from vyper.venom.analysis.available_expression import (
    NONIDEMPOTENT_INSTRUCTIONS,
    UNINTERESTING_OPCODES,
    CSEAnalysis,
)
from vyper.venom.analysis.dfg import DFGAnalysis
from vyper.venom.analysis.liveness import LivenessAnalysis
from vyper.venom.basicblock import IRInstruction, IRVariable
from vyper.venom.passes.base_pass import IRPass

# instruction that are not useful to be # substituted
NO_SUBSTITUTE_OPCODES = UNINTERESTING_OPCODES | frozenset(["offset"])


SMALL_EXPRESSION = 1


class CSE(IRPass):
    expression_analysis: CSEAnalysis

    def run_pass(self):
        available_expression_analysis = self.analyses_cache.request_analysis(CSEAnalysis)
        assert isinstance(available_expression_analysis, CSEAnalysis)
        self.expression_analysis = available_expression_analysis

        while True:
            replace_dict = self._find_replaceble()
            if len(replace_dict) == 0:
                return

            self._replace(replace_dict)
            self.analyses_cache.invalidate_analysis(DFGAnalysis)
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)
            self.expression_analysis = self.analyses_cache.force_analysis(CSEAnalysis)

    # return instruction and to which instruction it could
    # replaced by
    def _find_replaceble(self) -> dict[IRInstruction, IRInstruction]:
        res: dict[IRInstruction, IRInstruction] = dict()

        for bb in self.function.get_basic_blocks():
            for inst in bb.instructions:
                # skip instruction that for sure
                # wont be substituted
                if inst.opcode in NO_SUBSTITUTE_OPCODES:
                    continue
                if inst.opcode in NONIDEMPOTENT_INSTRUCTIONS:
                    continue
                expr, replace_inst = self.expression_analysis.get_expression(inst)
                if replace_inst == inst:
                    # no replacement
                    continue

                # heuristic to not replace small expressions across
                # basic block bounderies (it can create better codesize)
                if expr.depth > SMALL_EXPRESSION:
                    res[inst] = replace_inst
                else:
                    from_same_bb = self.expression_analysis.get_from_same_bb(inst, expr)
                    if len(from_same_bb) > 0 and from_same_bb[0] != inst:
                        # arbitrarily pick a replacement instruction
                        replace_inst = from_same_bb[0]
                        res[inst] = replace_inst

        return res

    def _replace(self, replace_dict: dict[IRInstruction, IRInstruction]):
        for orig, to in replace_dict.items():
            self._replace_inst(orig, to)

    def _replace_inst(self, orig_inst: IRInstruction, to_inst: IRInstruction):
        if orig_inst.output is not None:
            orig_inst.opcode = "store"
            assert isinstance(to_inst.output, IRVariable), f"not var {to_inst}"
            orig_inst.operands = [to_inst.output]
        else:
            orig_inst.opcode = "nop"
            orig_inst.operands = []
