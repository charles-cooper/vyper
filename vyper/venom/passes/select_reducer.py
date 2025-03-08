from vyper.venom.analysis import CFGAnalysis, DFGAnalysis, DominatorTreeAnalysis, LivenessAnalysis
from vyper.venom.basicblock import IRInstruction, IRLiteral, IROperand

# TODO: move InstructionUpdater to a more common file
from vyper.venom.passes.algebraic_optimization import InstructionUpdater
from vyper.venom.passes.base_pass import IRPass


class SelectReducer(IRPass):
    """
    This pass takes branches which select variables and converts them
    into branchless selects
    """

    def run_pass(self):
        self.doms = self.analyses_cache.request_analysis(DominatorTreeAnalysis)
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)

        for bb in self.function.get_basic_blocks():
            for inst in bb.instructions:
                if inst.opcode != "phi":
                    break
                self.handle_phi(inst)

            bb.ensure_well_formed()

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)
        self.analyses_cache.invalidate_analysis(DFGAnalysis)
        self.analyses_cache.invalidate_analysis(CFGAnalysis)

    def handle_phi(self, phi_inst: IRInstruction):
        bb = phi_inst.parent
        idom = self.doms.immediate_dominator(bb)
        if (jnz := idom.instructions[-1]).opcode != "jnz":
            return

        # heuristic: only apply if we think we can clear an entire bb
        for label, _var in phi_inst.phi_operands:
            source_bb = self.function.get_basic_block(label.value)
            if len(source_bb.instructions) > 2:
                return
            if not all(inst.opcode == "store" for inst in source_bb.instructions[:-1]):
                return

        ops = [var for _label, var in phi_inst.phi_operands]
        if len(ops) != 2:
            return

        source_vars: list[IROperand] = []
        for op in ops:
            source_var = self.dfg._traverse_store_chain(op)

            if isinstance(source_var, IRLiteral):
                source_vars.append(source_var)
                continue

            source_inst = self.dfg.get_producing_instruction(source_var)
            assert source_inst is not None, (op, source_var)  # help mypy
            if source_inst.parent != idom:
                return
            source_vars.append(source_var)

        updater = InstructionUpdater(self.dfg)
        cond = jnz.operands[0]
        inv_cond = updater._add_before(phi_inst, "iszero", [cond])
        xor = updater._add_before(phi_inst, "xor", source_vars)
        mul = updater._add_before(phi_inst, "mul", [inv_cond, xor])
        xor2 = updater._add_before(phi_inst, "xor", [source_vars[0], mul])

        updater._update(phi_inst, "store", [xor2])
