from collections import deque

from vyper.venom.analysis import DFGAnalysis, LivenessAnalysis
from vyper.venom.basicblock import IRInstruction, IRLabel, IROperand, IRVariable
from vyper.venom.passes.base_pass import IRPass
from vyper.venom.passes.machinery.inst_updater import InstUpdater


class InvokeArgCopyForwardingPass(IRPass):
    """
    Forward readonly memory invoke args through staging copies.

    Pattern:
      %tmp = alloca/calloca ...
      mcopy %tmp, %src, ...
      invoke @callee, ..., %tmp, ...

    For readonly callee memory params, `%tmp` can be replaced with `%src`.
    Dead mcopy/alloca cleanup is left to later DCE-style passes.
    """

    dfg: DFGAnalysis
    updater: InstUpdater

    def run_pass(self):
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.updater = InstUpdater(self.dfg)
        changed = False

        for bb in self.function.get_basic_blocks():
            for inst in list(bb.instructions):
                if inst.opcode != "mcopy":
                    continue
                changed |= self._try_forward_copy(inst)

        if changed:
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _try_forward_copy(self, copy_inst: IRInstruction) -> bool:
        dst = copy_inst.operands[2]
        if not isinstance(dst, IRVariable):
            return False

        root = self._assign_root_var(dst)
        root_inst = self.dfg.get_producing_instruction(root)
        if root_inst is None or root_inst.opcode not in ("alloca", "calloca"):
            return False

        aliases = self._collect_assign_aliases(root)
        rewrite_sites: set[tuple[IRInstruction, int]] = set()

        for var in aliases:
            for use in self.dfg.get_uses(var):
                positions = [i for i, op in enumerate(use.operands) if op == var]
                for pos in positions:
                    if use.opcode == "assign" and pos == 0:
                        continue
                    if use.opcode == "mcopy" and pos == 2:
                        if use is not copy_inst:
                            return False
                        continue
                    if use.opcode == "invoke" and self._is_readonly_invoke_operand(use, pos):
                        rewrite_sites.add((use, pos))
                        continue
                    return False

        if len(rewrite_sites) == 0:
            return False

        # Keep this local and conservative: only forward when all uses are
        # in the same block and dominated by the source copy.
        bb_insts = copy_inst.parent.instructions
        copy_idx = bb_insts.index(copy_inst)
        for invoke_inst, _ in rewrite_sites:
            if invoke_inst.parent is not copy_inst.parent:
                return False
            if bb_insts.index(invoke_inst) < copy_idx:
                return False

        src = self._assign_root(copy_inst.operands[1])
        if isinstance(src, IRVariable) and src in aliases:
            return False

        changed = False
        for invoke_inst, pos in rewrite_sites:
            if invoke_inst.operands[pos] == src:
                continue
            new_operands = list(invoke_inst.operands)
            new_operands[pos] = src
            self.updater.update(invoke_inst, invoke_inst.opcode, new_operands)
            changed = True

        if changed:
            self.updater.nop(copy_inst)
        return changed

    def _is_readonly_invoke_operand(self, invoke_inst: IRInstruction, operand_idx: int) -> bool:
        if operand_idx == 0:
            return False

        target = invoke_inst.operands[0]
        if not isinstance(target, IRLabel):
            return False

        try:
            callee = self.function.ctx.get_function(target)
        except Exception:
            return False

        readonly_idxs = callee._readonly_memory_invoke_arg_idxs
        return (operand_idx - 1) in readonly_idxs

    def _collect_assign_aliases(self, root: IRVariable) -> set[IRVariable]:
        aliases: set[IRVariable] = {root}
        worklist = deque([root])

        while len(worklist) > 0:
            var = worklist.popleft()
            for use in self.dfg.get_uses(var):
                if use.opcode != "assign":
                    continue
                out = use.output
                if out in aliases:
                    continue
                aliases.add(out)
                worklist.append(out)

        return aliases

    def _assign_root(self, op: IROperand) -> IROperand:
        if not isinstance(op, IRVariable):
            return op
        return self._assign_root_var(op)

    def _assign_root_var(self, var: IRVariable) -> IRVariable:
        while True:
            inst = self.dfg.get_producing_instruction(var)
            if inst is None or inst.opcode != "assign":
                return var
            parent = inst.operands[0]
            if not isinstance(parent, IRVariable):
                return var
            var = parent
