from vyper.venom.analysis import DFGAnalysis, LivenessAnalysis, DominatorTreeAnalysis, VarDefinition
from vyper.venom.check_venom import check_venom_fn
from vyper.venom.basicblock import IRVariable
from vyper.venom.passes.base_pass import InstUpdater, IRPass


class StoreElimination(IRPass):
    """
    This pass forwards variables to their uses though `store` instructions,
    and removes the `store` instruction.
    """

    # TODO: consider renaming `store` instruction, since it is confusing
    # with LoadElimination

    def run_pass(self):
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.dom = self.analyses_cache.request_analysis(DominatorTreeAnalysis)
        self.updater = InstUpdater(self.dfg)

        assert isinstance(self.dfg, DFGAnalysis)

        for var, inst in self.dfg.outputs.copy().items():
            if inst.opcode != "store":
                continue
            self._process_store(inst, var, inst.operands[0])

        for bb in self.function.get_basic_blocks():
            bb.ensure_well_formed()

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)
        self.analyses_cache.invalidate_analysis(VarDefinition)

    def _process_store(self, inst, var: IRVariable, new_var: IRVariable):
        """
        Process store instruction. If the variable is only used by a load instruction,
        forward the variable to the load instruction.
        """
        uses = self.dfg.get_uses(var)
        for use_inst in list(uses):
            if use_inst.opcode == "phi":
                new_source = self.dfg.get_producing_instruction(new_var)
                if new_source is None:
                    continue
                # invalid replacement
                if not self.dom.dominates(new_source.parent, inst.parent):
                    continue

            self.updater.update_operands(use_inst, {var: new_var})

            if use_inst.opcode == "phi":
                ops = [var for _, var in use_inst.phi_operands]
                if all(op == ops[0] for op in ops):
                    self.updater.store(use_inst, ops[0])

        if len(uses) == 0:
            self.updater.remove(inst)
