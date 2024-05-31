from collections import defaultdict
from dataclasses import asdict, dataclass

from vyper.utils import OrderedSet
from vyper.venom.analysis.dfg import DFGAnalysis
from vyper.venom.analysis.liveness import LivenessAnalysis
from vyper.venom.basicblock import IRBasicBlock, IRInstruction, IRVariable
from vyper.venom.function import IRFunction
from vyper.venom.passes.base_pass import IRPass

_ALL = ("storage", "transient", "memory", "immutables")

writes = {
    "sstore": "storage",
    "tstore": "transient",
    "mstore": "memory",
    "istore": "immutables",
    "delegatecall": _ALL,
    "call": _ALL,
    "create": _ALL,
    "create2": _ALL,
    "invoke": _ALL,  # could be smarter, look up the effects of the invoked function
    "staticcall": "memory",
    "dloadbytes": "memory",
    "returndatacopy": "memory",
    "calldatacopy": "memory",
    "codecopy": "memory",
    "extcodecopy": "memory",
    "mcopy": "memory",
}
reads = {
    "sload": "storage",
    "tload": "transient",
    "iload": "immutables",
    "mload": "memory",
    "mcopy": "memory",
    "call": _ALL,
    "delegatecall": _ALL,
    "staticcall": _ALL,
    "log": "memory",
    "revert": "memory",
    "return": "memory",
    "sha3": "memory",
}


@dataclass
class Fence:
    storage: int = 0
    memory: int = 0
    transient: int = 0
    immutables: int = 0


# effects graph
class EffectsG:
    def __init__(self):
        self._graph = defaultdict(list)

    def analyze(self, bb):
        fence = Fence()

        groups = {}
        terms = {}

        for inst in bb.instructions:
            reads = _get_reads(inst.opcode)
            writes = _get_writes(inst.opcode)
            for eff in reads:
                fence_id = getattr(fence, eff)
                group = groups.setdefault((eff, fence_id), [])
                group.append(inst)

            # collect writes in a separate dict
            for eff in writes:
                fence_id = getattr(fence, eff)
                assert (eff, fence_id) not in terms
                terms[(eff, fence_id)] = inst

            fence = _compute_fence(inst.opcode, fence)

        for (effect, fence_id), write_inst in terms.items():
            reads = groups.get((effect, fence_id), [])
            self._graph[write_inst].extend(reads)

            prev_id = fence_id - 1
            if (prev_write := terms.get((effect, prev_id))) is not None:
                self._graph[write_inst].append(prev_write)

            next_reads = groups.get((effect, fence_id + 1), [])
            for inst in next_reads:
                self._graph[inst].append(write_inst)

    def required_by(self, inst):
        return self._graph.get(inst, [])


def _get_reads(opcode):
    ret = reads.get(opcode, ())
    if not isinstance(ret, tuple):
        ret = (ret,)
    return ret


def _get_writes(opcode):
    ret = writes.get(opcode, ())
    if not isinstance(ret, tuple):
        ret = (ret,)
    return ret


def _compute_fence(opcode: str, fence: Fence) -> Fence:
    if opcode not in writes:
        return fence

    effects = _get_writes(opcode)

    tmp = asdict(fence)
    for eff in effects:
        tmp[eff] += 1

    return Fence(**tmp)


class DFTPass(IRPass):
    function: IRFunction
    fence: Fence

    def _process_instruction_r(self, bb: IRBasicBlock, inst: IRInstruction):
        for op in inst.get_outputs():
            assert isinstance(op, IRVariable), f"expected variable, got {op}"
            uses = self.dfg.get_uses(op)

            for uses_this in uses:
                if uses_this.parent != inst.parent:
                    continue

                self._process_instruction_r(bb, uses_this)

        if inst in self.visited_instructions:
            return
        self.visited_instructions.add(inst)

        for target in self._effects_g.required_by(inst):
            self._process_instruction_r(bb, target)

        for op in inst.get_input_variables():
            target = self.dfg.get_producing_instruction(op)
            assert target is not None, f"no producing instruction for {op}"
            if target.parent != inst.parent:
                continue
            self._process_instruction_r(bb, target)

        bb.instructions.append(inst)

    def _process_basic_block(self, bb: IRBasicBlock) -> None:
        self._effects_g = EffectsG()
        self._effects_g.analyze(bb)

        instructions = bb.instructions.copy()
        bb.instructions.clear()

        # start with out liveness
        for var in bb.out_vars:
            inst = self.dfg.get_producing_instruction(var)
            if inst.parent != bb:
                continue
            self._process_instruction_r(bb, inst)

        for inst in instructions:
            self._process_instruction_r(bb, inst)

        assert len(bb.instructions) == len(instructions), (instructions, bb)

        def key(inst):
            if inst.opcode == "phi":
                return 0
            if inst.is_bb_terminator:
                return 2
            return 1

        bb.instructions.sort(key=key)

    def run_pass(self) -> None:
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.analyses_cache.request_analysis(LivenessAnalysis)  # use out_vars

        self.fence = Fence()
        self.visited_instructions: OrderedSet[IRInstruction] = OrderedSet()

        for bb in self.function.get_basic_blocks():
            self._process_basic_block(bb)

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)
