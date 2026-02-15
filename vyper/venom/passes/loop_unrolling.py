from __future__ import annotations

from dataclasses import dataclass

from vyper.venom.analysis import CFGAnalysis, DFGAnalysis
from vyper.venom.basicblock import IRBasicBlock, IRInstruction, IRLabel, IRLiteral, IROperand, IRVariable
from vyper.venom.passes.base_pass import IRPass

MAX_FULL_UNROLL_ITERATIONS = 8


@dataclass(frozen=True)
class _SSALoopShape:
    preheader: IRBasicBlock
    header: IRBasicBlock
    body: IRBasicBlock
    exit: IRBasicBlock
    phi: IRInstruction
    backedge_value: IRVariable
    end_operand: IROperand
    exact_trip_count: int | None
    max_trip_count: int | None


class LoopUnrollingPass(IRPass):
    """
    Fully unroll canonical SSA counted loops with a small compile-time bound.

    SSA shape:
    - header has a single phi `%counter = phi @preheader, %start, @body, %next`
    - header condition is based on `xor %counter, %end` (directly or via `iszero`)
    - body has a single successor back to header and defines `%next`

    For SSA loops with unknown trip count, we also support guarded full-unroll
    when a compile-time upper bound is proven by an assertion in the preheader.
    """

    required_successors = ("SimplifyCFGPass",)

    def run_pass(self, /, max_full_unroll_iterations: int = MAX_FULL_UNROLL_ITERATIONS) -> None:
        if max_full_unroll_iterations <= 0:
            return

        changed = False
        # Recompute CFG after each rewrite so subsequent matches use fresh topology.
        while True:
            cfg = self.analyses_cache.force_analysis(CFGAnalysis)
            dfg = self.analyses_cache.force_analysis(DFGAnalysis)
            rewritten = False

            for cond in list(self.function.get_basic_blocks()):
                if not cfg.is_reachable(cond):
                    continue

                ssa_loop = self._match_ssa_loop(cfg, dfg, cond)
                if ssa_loop is None:
                    continue

                exact_trip_count = ssa_loop.exact_trip_count
                if exact_trip_count is not None:
                    if exact_trip_count > max_full_unroll_iterations:
                        continue
                    self._unroll_ssa_exact(ssa_loop, exact_trip_count)
                    rewritten = True
                    changed = True
                    break

                max_trip_count = ssa_loop.max_trip_count
                if max_trip_count is None or max_trip_count > max_full_unroll_iterations:
                    continue

                self._unroll_ssa_guarded(ssa_loop, max_trip_count)
                rewritten = True
                changed = True
                break

            if not rewritten:
                break

        if changed:
            self.analyses_cache.invalidate_analysis(CFGAnalysis)
            self.analyses_cache.invalidate_analysis(DFGAnalysis)

    def _match_ssa_loop(
        self, cfg: CFGAnalysis, dfg: DFGAnalysis, header: IRBasicBlock
    ) -> _SSALoopShape | None:
        phi_insts = list(header.phi_instructions)
        if len(phi_insts) != 1:
            return None

        term = header.last_instruction
        if term.opcode != "jnz" or len(term.operands) != 3:
            return None

        cond_operand, succ1_label, succ2_label = term.operands
        if not isinstance(cond_operand, IRVariable):
            return None
        if not isinstance(succ1_label, IRLabel) or not isinstance(succ2_label, IRLabel):
            return None

        fn = self.function
        succ1 = fn.get_basic_block(succ1_label.value)
        succ2 = fn.get_basic_block(succ2_label.value)

        body: IRBasicBlock | None = None
        exit_bb: IRBasicBlock | None = None
        if len(cfg.cfg_out(succ1)) == 1 and header in cfg.cfg_out(succ1):
            body, exit_bb = succ1, succ2
        elif len(cfg.cfg_out(succ2)) == 1 and header in cfg.cfg_out(succ2):
            body, exit_bb = succ2, succ1
        if body is None or exit_bb is None:
            return None

        if self._has_phi(body) or self._has_phi(exit_bb):
            return None
        if body.last_instruction.opcode != "jmp":
            return None
        if len(body.last_instruction.operands) != 1 or body.last_instruction.operands[0] != header.label:
            return None

        if len(cfg.cfg_out(header)) != 2:
            return None
        if len(cfg.cfg_in(header)) != 2 or body not in cfg.cfg_in(header):
            return None
        if len(cfg.cfg_in(body)) != 1 or header not in cfg.cfg_in(body):
            return None
        if len(cfg.cfg_out(body)) != 1 or header not in cfg.cfg_out(body):
            return None

        phi = phi_insts[0]
        if len(phi.operands) != 4:
            return None

        # If phi output escapes the loop, replacement needs exit-path-specific
        # materialization and is currently unsupported.
        for use_inst in dfg.get_uses(phi.output):
            if use_inst.parent not in (header, body):
                return None

        preheader: IRBasicBlock | None = None
        start_value: IRVariable | None = None
        backedge_value: IRVariable | None = None
        for label, value in phi.phi_operands:
            if label == body.label:
                backedge_value = value
            else:
                preheader = fn.get_basic_block(label.value)
                start_value = value

        if preheader is None or start_value is None or backedge_value is None:
            return None
        if preheader.last_instruction.opcode != "jmp":
            return None
        if len(preheader.last_instruction.operands) != 1 or preheader.last_instruction.operands[0] != header.label:
            return None
        if len(cfg.cfg_out(preheader)) != 1 or header not in cfg.cfg_out(preheader):
            return None

        if not self._is_var_defined_in_block(backedge_value, body):
            return None

        end_operand = self._extract_header_end_operand(header, phi.output, cond_operand)
        if end_operand is None:
            return None

        pre_defs = self._build_local_defs(preheader)

        start_literal = self._try_eval_literal(start_value, pre_defs)
        end_literal = self._try_eval_literal(end_operand, pre_defs)

        exact_trip_count: int | None = None
        max_trip_count: int | None = None

        if start_literal is not None and end_literal is not None:
            trip_count = end_literal - start_literal
            if trip_count < 0:
                return None
            exact_trip_count = trip_count
        else:
            max_trip_count = self._try_get_max_trip_bound(preheader, pre_defs, end_operand)

        if exact_trip_count is None and max_trip_count is None:
            return None

        return _SSALoopShape(
            preheader=preheader,
            header=header,
            body=body,
            exit=exit_bb,
            phi=phi,
            backedge_value=backedge_value,
            end_operand=end_operand,
            exact_trip_count=exact_trip_count,
            max_trip_count=max_trip_count,
        )

    def _unroll_ssa_exact(self, loop: _SSALoopShape, trip_count: int) -> None:
        pre_term = loop.preheader.last_instruction

        if trip_count == 0:
            pre_term.operands = [loop.exit.label]
            self.function.remove_basic_block(loop.header)
            self.function.remove_basic_block(loop.body)
            return

        state_var = self._preheader_phi_value(loop)

        unrolled: list[IRBasicBlock] = []
        for _ in range(trip_count):
            label = self.function.ctx.get_next_label(f"unroll_{loop.body.label.value}")
            bb = IRBasicBlock(label, self.function)

            cloned, subst = self._clone_ssa_instructions(
                loop.body.instructions[:-1], {loop.phi.output: state_var}
            )
            for inst in cloned:
                bb.insert_instruction(inst)

            next_state = subst.get(loop.backedge_value)
            assert isinstance(next_state, IRVariable)

            self.function.append_basic_block(bb)
            unrolled.append(bb)
            state_var = next_state

        pre_term.operands = [unrolled[0].label]
        for idx, bb in enumerate(unrolled):
            next_label = loop.exit.label if idx == len(unrolled) - 1 else unrolled[idx + 1].label
            bb.append_instruction("jmp", next_label)

        self.function.remove_basic_block(loop.header)
        self.function.remove_basic_block(loop.body)

    def _unroll_ssa_guarded(self, loop: _SSALoopShape, max_trip_count: int) -> None:
        pre_term = loop.preheader.last_instruction

        if max_trip_count == 0:
            pre_term.operands = [loop.exit.label]
            self.function.remove_basic_block(loop.header)
            self.function.remove_basic_block(loop.body)
            return

        state_var = self._preheader_phi_value(loop)
        header_term = loop.header.last_instruction
        assert header_term.opcode == "jnz"
        old_true_label = header_term.operands[1]
        old_false_label = header_term.operands[2]
        assert isinstance(old_true_label, IRLabel)
        assert isinstance(old_false_label, IRLabel)

        if loop.body.label == old_true_label:
            branch_to_body_is_true = True
        elif loop.body.label == old_false_label:
            branch_to_body_is_true = False
        else:
            raise AssertionError("invalid loop shape: body label is not a header successor")

        cond_blocks: list[IRBasicBlock] = []
        body_blocks: list[IRBasicBlock] = []

        for i in range(max_trip_count):
            cond_label = self.function.ctx.get_next_label(f"unroll_cond_{loop.body.label.value}_{i}")
            body_label = self.function.ctx.get_next_label(f"unroll_body_{loop.body.label.value}_{i}")

            cond_bb = IRBasicBlock(cond_label, self.function)
            body_bb = IRBasicBlock(body_label, self.function)

            cond_cloned, cond_subst = self._clone_ssa_instructions(
                self._header_non_phi_non_terminators(loop.header),
                {loop.phi.output: state_var},
            )
            for inst in cond_cloned:
                cond_bb.insert_instruction(inst)

            cond_var = header_term.operands[0]
            if isinstance(cond_var, IRVariable):
                cond_var = cond_subst.get(cond_var, cond_var)

            if branch_to_body_is_true:
                cond_bb.append_instruction("jnz", cond_var, body_label, loop.exit.label)
            else:
                cond_bb.append_instruction("jnz", cond_var, loop.exit.label, body_label)

            body_cloned, body_subst = self._clone_ssa_instructions(
                loop.body.instructions[:-1], cond_subst
            )
            for inst in body_cloned:
                body_bb.insert_instruction(inst)

            next_state = body_subst.get(loop.backedge_value)
            assert isinstance(next_state, IRVariable)
            state_var = next_state

            self.function.append_basic_block(cond_bb)
            self.function.append_basic_block(body_bb)
            cond_blocks.append(cond_bb)
            body_blocks.append(body_bb)

        pre_term.operands = [cond_blocks[0].label]

        for idx, body_bb in enumerate(body_blocks):
            next_label = loop.exit.label if idx == len(body_blocks) - 1 else cond_blocks[idx + 1].label
            body_bb.append_instruction("jmp", next_label)

        self.function.remove_basic_block(loop.header)
        self.function.remove_basic_block(loop.body)

    def _has_phi(self, bb: IRBasicBlock) -> bool:
        return any(inst.opcode == "phi" for inst in bb.instructions)

    def _is_var_defined_in_block(self, var: IRVariable, bb: IRBasicBlock) -> bool:
        for inst in bb.instructions:
            if var in inst.get_outputs():
                return True
        return False

    def _preheader_phi_value(self, loop: _SSALoopShape) -> IRVariable:
        for label, value in loop.phi.phi_operands:
            if label == loop.preheader.label:
                return value
        raise AssertionError("invalid loop shape: preheader operand missing in phi")

    def _header_non_phi_non_terminators(self, header: IRBasicBlock) -> list[IRInstruction]:
        ret: list[IRInstruction] = []
        for inst in header.instructions:
            if inst.opcode == "phi":
                continue
            if inst.is_bb_terminator:
                break
            ret.append(inst)
        return ret

    def _clone_ssa_instructions(
        self, instructions: list[IRInstruction], subst: dict[IRVariable, IRVariable]
    ) -> tuple[list[IRInstruction], dict[IRVariable, IRVariable]]:
        new_subst = dict(subst)
        cloned: list[IRInstruction] = []

        for inst in instructions:
            new_inst = inst.copy()

            new_operands: list[IROperand] = []
            for op in new_inst.operands:
                if isinstance(op, IRVariable) and op in new_subst:
                    new_operands.append(new_subst[op])
                else:
                    new_operands.append(op)
            new_inst.operands = new_operands

            if new_inst.has_outputs:
                new_outputs: list[IRVariable] = []
                for old_out in inst.get_outputs():
                    fresh = self.function.get_next_variable()
                    new_subst[old_out] = fresh
                    new_outputs.append(fresh)
                new_inst.set_outputs(new_outputs)

            cloned.append(new_inst)

        return cloned, new_subst

    def _extract_header_end_operand(
        self, header: IRBasicBlock, counter_var: IRVariable, cond_operand: IRVariable
    ) -> IROperand | None:
        produced: dict[IRVariable, IRInstruction] = {}
        for inst in header.instructions:
            for out in inst.get_outputs():
                produced[out] = inst

        cond_inst = produced.get(cond_operand)
        xor_inst: IRInstruction | None = None
        if cond_inst is None:
            return None
        if cond_inst.opcode == "xor":
            xor_inst = cond_inst
        elif (
            cond_inst.opcode == "iszero"
            and len(cond_inst.operands) == 1
            and isinstance(cond_inst.operands[0], IRVariable)
        ):
            xor_inst = produced.get(cond_inst.operands[0])

        if xor_inst is None or xor_inst.opcode != "xor" or len(xor_inst.operands) != 2:
            return None

        op1, op2 = xor_inst.operands
        if op1 == counter_var and op2 != counter_var:
            return op2
        if op2 == counter_var and op1 != counter_var:
            return op1

        return None

    def _build_local_defs(self, bb: IRBasicBlock) -> dict[IRVariable, IRInstruction]:
        defs: dict[IRVariable, IRInstruction] = {}
        for inst in bb.instructions:
            if inst.is_bb_terminator:
                break
            for output in inst.get_outputs():
                defs[output] = inst
        return defs

    def _try_eval_literal(
        self, operand: IROperand, defs: dict[IRVariable, IRInstruction], depth: int = 0
    ) -> int | None:
        if isinstance(operand, IRLiteral):
            return operand.value
        if not isinstance(operand, IRVariable):
            return None
        if depth > 32:
            return None

        inst = defs.get(operand)
        if inst is None:
            return None

        if inst.opcode == "assign" and len(inst.operands) == 1:
            return self._try_eval_literal(inst.operands[0], defs, depth + 1)

        if inst.opcode == "add" and len(inst.operands) == 2:
            left = self._try_eval_literal(inst.operands[0], defs, depth + 1)
            right = self._try_eval_literal(inst.operands[1], defs, depth + 1)
            if left is None or right is None:
                return None
            return left + right

        return None

    def _resolve_var(
        self, operand: IROperand, defs: dict[IRVariable, IRInstruction], depth: int = 0
    ) -> IROperand:
        if not isinstance(operand, IRVariable) or depth > 32:
            return operand

        inst = defs.get(operand)
        if inst is None or inst.opcode != "assign" or len(inst.operands) != 1:
            return operand

        return self._resolve_var(inst.operands[0], defs, depth + 1)

    def _try_get_max_trip_bound(
        self, preheader: IRBasicBlock, defs: dict[IRVariable, IRInstruction], end_operand: IROperand
    ) -> int | None:
        if not isinstance(end_operand, IRVariable):
            return None

        resolved_end = self._resolve_var(end_operand, defs)
        if not isinstance(resolved_end, IRVariable):
            return None

        end_inst = defs.get(resolved_end)
        if end_inst is None or end_inst.opcode != "add" or len(end_inst.operands) != 2:
            return None

        add_vars = {
            self._resolve_var(op, defs)
            for op in end_inst.operands
            if isinstance(self._resolve_var(op, defs), IRVariable)
        }
        if len(add_vars) == 0:
            return None

        for inst in preheader.instructions:
            if inst.opcode != "assert" or len(inst.operands) != 1:
                continue

            assert_arg = self._resolve_var(inst.operands[0], defs)
            if not isinstance(assert_arg, IRVariable):
                continue

            iszero_inst = defs.get(assert_arg)
            if (
                iszero_inst is None
                or iszero_inst.opcode != "iszero"
                or len(iszero_inst.operands) != 1
                or not isinstance(iszero_inst.operands[0], IRVariable)
            ):
                continue

            cmp_inst = defs.get(iszero_inst.operands[0])
            if cmp_inst is None or cmp_inst.opcode not in ("gt", "lt") or len(cmp_inst.operands) != 2:
                continue

            first = self._resolve_var(cmp_inst.operands[-1], defs)
            second = self._resolve_var(cmp_inst.operands[-2], defs)
            first_lit = self._try_eval_literal(first, defs)
            second_lit = self._try_eval_literal(second, defs)

            trip_var: IRVariable | None = None
            bound: int | None = None

            # iszero(gt trip, bound)) -> trip <= bound
            if cmp_inst.opcode == "gt" and isinstance(first, IRVariable) and second_lit is not None:
                trip_var = first
                bound = second_lit

            # iszero(lt bound, trip)) -> trip <= bound
            if cmp_inst.opcode == "lt" and isinstance(second, IRVariable) and first_lit is not None:
                trip_var = second
                bound = first_lit

            if trip_var is None or bound is None or bound < 0:
                continue

            resolved_trip = self._resolve_var(trip_var, defs)
            if not isinstance(resolved_trip, IRVariable):
                continue

            if resolved_trip in add_vars:
                return bound

        return None
