from collections import defaultdict

import vyper.venom.effects as effects
from vyper.utils import OrderedSet
from vyper.venom.analysis import DFGAnalysis, LivenessAnalysis
from vyper.venom.basicblock import IRBasicBlock, IRInstruction
from vyper.venom.function import IRFunction
from vyper.venom.passes.base_pass import IRPass


class DFTPass(IRPass):
    function: IRFunction
    data_offspring: dict[IRInstruction, OrderedSet[IRInstruction]]
    visited_instructions: OrderedSet[IRInstruction]
    # "data dependency analysis"
    dda: dict[IRInstruction, OrderedSet[IRInstruction]]
    # "effect dependency analysis"
    eda: dict[IRInstruction, OrderedSet[IRInstruction]]

    def run_pass(self) -> None:
        self.data_offspring = {}
        self.visited_instructions: OrderedSet[IRInstruction] = OrderedSet()

        self.dfg = self.analyses_cache.force_analysis(DFGAnalysis)
        self.liveness = self.analyses_cache.force_analysis(LivenessAnalysis)

        self.last_changed = defaultdict(int)

        worklist = OrderedSet(self.function.get_basic_blocks())
        while len(worklist) > 0:
            bb = worklist.pop()
            changed = self._process_basic_block(bb)
            if changed:
                worklist.update(bb.cfg_in)

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _process_basic_block(self, bb: IRBasicBlock) -> None:
        self.liveness._calculate_out_vars(bb)

        self._calculate_dependency_graphs(bb)
        self.instructions = list(bb.pseudo_instructions)
        non_phi_instructions = list(bb.non_phi_instructions)

        self.visited_instructions = OrderedSet()
        for inst in bb.instructions:
            self._calculate_data_offspring(inst)

        entry_instructions: OrderedSet[IRInstruction] = OrderedSet()
        for var in bb.out_vars:
            src = self.dfg.get_producing_instruction(var)
            assert src is not None
            if src.parent != bb:
                continue
            entry_instructions.add(src)

        # Compute entry points in the graph of instruction dependencies
        entry_instructions.update(non_phi_instructions)
        for inst in non_phi_instructions:
            to_remove = self.dda.get(inst, OrderedSet()) | self.eda.get(inst, OrderedSet())
            entry_instructions.dropmany(to_remove)

        entry_instructions_list = list(entry_instructions)

        self.visited_instructions = OrderedSet()
        for inst in entry_instructions_list:
            self._process_instruction_r(self.instructions, inst)

        changed = sum(x != y for (x,y) in zip(bb.instructions, self.instructions))
        # minimize changes
        if changed > self.last_changed[bb]:
            return
        self.last_changed[bb] = changed
        self.liveness._calculate_liveness(bb)  # recompute

        bb.instructions = self.instructions
        assert bb.is_terminated, f"Basic block should be terminated {bb}"

        return changed

    def _process_instruction_r(self, instructions: list[IRInstruction], inst: IRInstruction):
        if inst in self.visited_instructions:
            return
        self.visited_instructions.add(inst)

        if inst.is_pseudo:
            return

        children = list(self.dda[inst] | self.eda[inst])

        def cost(x: IRInstruction) -> int | float:
            if x in self.eda[inst] or inst.flippable:
                ret = -1 * int(len(self.data_offspring[x]) > 0)
            else:
                assert x in self.dda[inst]  # sanity check
                assert x.output is not None  # help mypy
                ret = inst.operands.index(x.output)
            return ret

        # heuristic: sort by size of child dependency graph
        orig_children = children.copy()
        children.sort(key=cost)

        if inst.flippable and (orig_children != children):
            inst.flip()

        for dep_inst in children:
            self._process_instruction_r(instructions, dep_inst)

        instructions.append(inst)

    def _calculate_dependency_graphs(self, bb: IRBasicBlock) -> None:
        # ida: instruction dependency analysis
        self.dda = defaultdict(OrderedSet)
        self.eda = defaultdict(OrderedSet)

        non_phis = list(bb.non_phi_instructions)

        #
        # Compute dependency graph
        #
        last_write_effects: dict[effects.Effects, IRInstruction] = {}
        last_read_effects: dict[effects.Effects, IRInstruction] = {}

        for inst in non_phis:
            for op in inst.operands:
                dep = self.dfg.get_producing_instruction(op)
                if dep is not None and dep.parent == bb:
                    self.dda[inst].add(dep)

            write_effects = inst.get_write_effects()
            read_effects = inst.get_read_effects()

            for write_effect in write_effects:
                if write_effect in last_read_effects:
                    self.eda[inst].add(last_read_effects[write_effect])
                last_write_effects[write_effect] = inst

            for read_effect in read_effects:
                if read_effect in last_write_effects and last_write_effects[read_effect] != inst:
                    self.eda[inst].add(last_write_effects[read_effect])
                last_read_effects[read_effect] = inst

    def _calculate_data_offspring(self, inst: IRInstruction):
        if inst in self.data_offspring:
            return self.data_offspring[inst]

        self.data_offspring[inst] = self.dda[inst].copy()

        deps = self.dda[inst]
        for dep_inst in deps:
            assert inst.parent == dep_inst.parent
            res = self._calculate_data_offspring(dep_inst)
            self.data_offspring[inst] |= res

        return self.data_offspring[inst]
