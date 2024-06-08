from vyper.utils import OrderedSet
from vyper.venom.analysis.cfg import CFGAnalysis
from vyper.venom.analysis.dfg import DFGAnalysis
from vyper.venom.analysis.liveness import LivenessAnalysis
from vyper.venom.basicblock import IRBasicBlock, IRInstruction, IRVariable
from vyper.venom.function import IRFunction
from vyper.venom.passes.base_pass import IRPass


class Mem2Var(IRPass):
    """
    This pass promoted memory operations to variable operations, when possible.
    It does yet do any memory aliasing analysis, so it is conservative.
    """

    function: IRFunction
    defs: dict[IRVariable, OrderedSet[IRBasicBlock]]

    def run_pass(self):
        self.analyses_cache.request_analysis(CFGAnalysis)
        dfg = self.analyses_cache.request_analysis(DFGAnalysis)

        self.var_name_count = 0
        for var, inst in dfg.outputs.items():
            if inst.opcode != "alloca":
                continue
            self._process_alloca_var(dfg, var)

        self.analyses_cache.invalidate_analysis(DFGAnalysis)
        self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _process_alloca_var(self, dfg: DFGAnalysis, var: IRVariable):
        """
        Process alloca allocated variable. If it is only used by mstore/mload/return
        instructions, it is promoted to a stack variable. Otherwise, it is left as is.
        """
        uses = dfg.get_uses(var)
        if all([inst.opcode == "mload" for inst in uses]):
            return
        #elif all([inst.opcode == "mstore" for inst in uses]):
        #    return
        elif all(_ok_use(inst) for inst in uses):
            var_name = f"addr{var.name}_{self.var_name_count}"
            self.var_name_count += 1
            for inst in uses:
                if inst.opcode == "mstore":
                    inst.opcode = "store"
                    inst.output = IRVariable(var_name)
                    inst.operands = [inst.operands[0]]
                elif inst.opcode == "mload":
                    inst.opcode = "store"
                    inst.operands = [IRVariable(var_name)]
                elif inst.opcode == "return" or inst.opcode=="store"and inst.output.value=="%ret_ofst":
                    bb = inst.parent
                    idx = bb.instructions.index(inst)
                    op_ix = 1 if inst.opcode=="return" else 0
                    val = inst.operands[op_ix]
                    bb.insert_instruction(
                        IRInstruction("mstore", [IRVariable(var_name), val]), idx
                    )

def _ok_use(inst):
    if inst.opcode in ("mstore", "mload", "return"):
        return True
    return inst.opcode == "store" and inst.output.value == "%ret_ofst"
