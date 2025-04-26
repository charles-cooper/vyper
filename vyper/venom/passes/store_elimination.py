from vyper.venom.analysis import DFGAnalysis, LivenessAnalysis, DominatorTreeAnalysis
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

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _process_store(self, inst, var: IRVariable, new_var: IRVariable):
        """
        Process store instruction. If the variable is only used by a load instruction,
        forward the variable to the load instruction.
        """
        uses = self.dfg.get_uses(var)
        for use_inst in list(uses):
            if use_inst.opcode == "phi":
                new_src = self.dfg.get_producing_instruction(new_var)
                valid_replacement = new_src is not None and self.dom.dominates(new_src.parent, use_inst.parent)
                if not valid_replacement:
                    continue
            self.updater.update_operands(use_inst, {var: new_var})

        if len(uses) == 0:
            self.updater.remove(inst)
