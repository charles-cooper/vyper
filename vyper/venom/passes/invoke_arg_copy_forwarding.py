from collections import deque

import vyper.evm.address_space as addr_space
from vyper.venom.analysis import (
    BasePtrAnalysis,
    DFGAnalysis,
    DominatorTreeAnalysis,
    LivenessAnalysis,
    MemoryAliasAnalysis,
)
from vyper.venom.basicblock import IRInstruction, IRLabel, IRLiteral, IROperand, IRVariable
from vyper.venom.effects import EMPTY, Effects
from vyper.venom.passes.base_pass import IRPass
from vyper.venom.passes.machinery.inst_updater import InstUpdater


class InvokeArgCopyForwardingPass(IRPass):
    """
    Forward memory copies in two conservative invoke-related patterns.

    1) Readonly invoke-arg forwarding:
      %tmp = alloca/calloca ...
      mcopy %tmp, %src, ...
      invoke @callee, ..., %tmp, ...

    For readonly callee memory params, `%tmp` can be replaced with `%src`.

    2) Internal-call return-buffer forwarding:
      invoke @callee_returning_memory, %ret_buf, ...
      mcopy %dst, %ret_buf, ...

    If `%ret_buf` is only consumed by that invoke and this copy, and `%dst`
    is only used after the copy, rewrite `%dst` uses to `%ret_buf`.
    Dead mcopy/alloca cleanup is left to later DCE-style passes.
    """

    dfg: DFGAnalysis
    domtree: DominatorTreeAnalysis
    base_ptr: BasePtrAnalysis
    mem_alias: MemoryAliasAnalysis
    updater: InstUpdater

    def run_pass(self):
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.domtree = self.analyses_cache.request_analysis(DominatorTreeAnalysis)
        self.base_ptr = self.analyses_cache.request_analysis(BasePtrAnalysis)
        self.mem_alias = self.analyses_cache.request_analysis(MemoryAliasAnalysis)
        self.updater = InstUpdater(self.dfg)
        changed = False

        for bb in self.function.get_basic_blocks():
            for inst in list(bb.instructions):
                if inst.opcode != "mcopy":
                    continue
                if self._try_forward_internal_return_copy(inst):
                    changed = True
                    continue
                changed |= self._try_forward_readonly_copy(inst)

        if changed:
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _try_forward_internal_return_copy(self, copy_inst: IRInstruction) -> bool:
        dst = copy_inst.operands[2]
        src = copy_inst.operands[1]
        size = copy_inst.operands[0]
        if not isinstance(dst, IRVariable) or not isinstance(src, IRVariable):
            return False
        if not isinstance(size, IRLiteral):
            return False

        dst_root = self._assign_root_var(dst)
        src_root = self._assign_root_var(src)
        if dst_root == src_root:
            return False

        dst_root_inst = self.dfg.get_producing_instruction(dst_root)
        src_root_inst = self.dfg.get_producing_instruction(src_root)

        if not self._is_alloca_like(dst_root_inst) or not self._is_alloca_like(src_root_inst):
            return False
        assert dst_root_inst is not None and src_root_inst is not None  # ensured above
        if not self._matches_alloca_size(dst_root_inst, size.value):
            return False
        if not self._matches_alloca_size(src_root_inst, size.value):
            return False

        if not self._is_internal_return_buffer_source(src_root, copy_inst):
            return False

        dst_aliases = self._collect_assign_aliases(dst_root)
        rewrite_insts: set[IRInstruction] = set()

        for var in dst_aliases:
            for use in self.dfg.get_uses(var):
                positions = [i for i, op in enumerate(use.operands) if op == var]
                for pos in positions:
                    if use.opcode == "assign" and pos == 0:
                        continue
                    if use.opcode == "mcopy" and pos == 2 and use is copy_inst:
                        continue
                    if use.opcode == "phi":
                        return False
                    if not self._is_after(copy_inst, use):
                        return False
                    rewrite_insts.add(use)

        replace_map: dict[IROperand, IROperand] = {var: src_root for var in dst_aliases}
        for use in rewrite_insts:
            self.updater.update_operands(use, replace_map)

        self.updater.nop(copy_inst)
        return True

    def _try_forward_readonly_copy(self, copy_inst: IRInstruction) -> bool:
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

        if self._has_src_clobber_between(copy_inst, rewrite_sites):
            return False

        src = self._assign_root(copy_inst.operands[1])
        if isinstance(src, IRVariable) and src in aliases:
            return False
        if isinstance(src, IRVariable) and self._has_mutable_same_source_sibling_arg(
            rewrite_sites, src
        ):
            return False

        for invoke_inst, pos in rewrite_sites:
            if invoke_inst.operands[pos] == src:
                continue
            new_operands = list(invoke_inst.operands)
            new_operands[pos] = src
            self.updater.update(invoke_inst, invoke_inst.opcode, new_operands)

        # Even when operands already point to src, this copy is redundant:
        # all remaining uses are readonly invokes validated above.
        self.updater.nop(copy_inst)
        return True

    def _has_mutable_same_source_sibling_arg(
        self, rewrite_sites: set[tuple[IRInstruction, int]], src_root: IRVariable
    ) -> bool:
        """
        Reject readonly forwarding when it would create aliasing between a
        rewritten readonly arg and a sibling mutable arg in the same invoke:

            invoke @f, %tmp_ro, ..., %src_mut
            # %tmp_ro came from mcopy(%tmp_ro <- %src_root)

        Rewriting %tmp_ro -> %src_root would change call semantics to pass the
        same memory region to both params.
        """
        for invoke_inst, rewritten_pos in rewrite_sites:
            for pos, op in enumerate(invoke_inst.operands):
                if pos == 0 or pos == rewritten_pos:
                    continue
                if self._is_readonly_invoke_operand(invoke_inst, pos):
                    continue
                root = self._assign_root(op)
                if root == src_root:
                    return True
        return False

    def _is_after(self, copy_inst: IRInstruction, use_inst: IRInstruction) -> bool:
        copy_bb = copy_inst.parent
        use_bb = use_inst.parent

        if use_bb is copy_bb:
            bb_insts = copy_bb.instructions
            return bb_insts.index(use_inst) > bb_insts.index(copy_inst)

        return self.domtree.dominates(copy_bb, use_bb)

    def _is_internal_return_buffer_source(
        self, src_root: IRVariable, copy_inst: IRInstruction
    ) -> bool:
        aliases = self._collect_assign_aliases(src_root)
        copy_seen = False
        invoke_sites: set[IRInstruction] = set()

        copy_bb = copy_inst.parent
        bb_insts = copy_bb.instructions
        copy_idx = bb_insts.index(copy_inst)

        for var in aliases:
            for use in self.dfg.get_uses(var):
                positions = [i for i, op in enumerate(use.operands) if op == var]
                for pos in positions:
                    if use.opcode == "assign" and pos == 0:
                        continue

                    if use.opcode == "mcopy" and pos == 1 and use is copy_inst:
                        copy_seen = True
                        continue

                    if use.opcode == "invoke" and pos == 1:
                        if use.parent is not copy_bb:
                            return False
                        if bb_insts.index(use) >= copy_idx:
                            return False
                        if not self._invoke_has_return_buffer(use):
                            return False
                        invoke_sites.add(use)
                        continue

                    return False

        return copy_seen and len(invoke_sites) == 1

    def _invoke_has_return_buffer(self, invoke_inst: IRInstruction) -> bool:
        target = invoke_inst.operands[0]
        if not isinstance(target, IRLabel):
            return False

        callee = self.function.ctx.functions.get(target)
        if callee is None:
            return False

        if callee._invoke_param_count is None or callee._has_memory_return_buffer_param is None:
            return False

        invoke_arg_count = len(invoke_inst.operands) - 1
        if invoke_arg_count != callee._invoke_param_count:
            return False

        return callee._has_memory_return_buffer_param

    def _is_alloca_like(self, inst: IRInstruction | None) -> bool:
        return inst is not None and inst.opcode in ("alloca", "calloca")

    def _matches_alloca_size(self, inst: IRInstruction, expected_size: int) -> bool:
        size = inst.operands[0]
        return isinstance(size, IRLiteral) and size.value == expected_size

    def _is_readonly_invoke_operand(self, invoke_inst: IRInstruction, operand_idx: int) -> bool:
        if operand_idx == 0:
            return False

        target = invoke_inst.operands[0]
        if not isinstance(target, IRLabel):
            return False

        callee = self.function.ctx.functions.get(target)
        if callee is None:
            return False

        readonly_idxs = callee._readonly_memory_invoke_arg_idxs
        return (operand_idx - 1) in readonly_idxs

    def _has_src_clobber_between(
        self, copy_inst: IRInstruction, rewrite_sites: set[tuple[IRInstruction, int]]
    ) -> bool:
        src_loc = self.base_ptr.get_read_location(copy_inst, addr_space.MEMORY)
        if src_loc.is_empty():
            return False

        bb_insts = copy_inst.parent.instructions
        copy_idx = bb_insts.index(copy_inst)

        for invoke_inst, _ in rewrite_sites:
            invoke_idx = bb_insts.index(invoke_inst)
            for inst in bb_insts[copy_idx + 1 : invoke_idx]:
                if inst.get_write_effects() & Effects.MEMORY == EMPTY:
                    continue
                write_loc = self.base_ptr.get_write_location(inst, addr_space.MEMORY)
                if self.mem_alias.may_alias(src_loc, write_loc):
                    return True

        return False

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
