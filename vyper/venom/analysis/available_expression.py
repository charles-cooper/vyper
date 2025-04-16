from __future__ import annotations
from vyper.utils import profileit, cumtimeit

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import vyper.venom.effects as effects
from vyper.venom.analysis.analysis import IRAnalysesCache, IRAnalysis
from vyper.venom.analysis.cfg import CFGAnalysis
from vyper.venom.analysis.dfg import DFGAnalysis
from vyper.venom.basicblock import (
    BB_TERMINATORS,
    COMMUTATIVE_INSTRUCTIONS,
    IRBasicBlock,
    IRInstruction,
    IROperand,
    IRVariable,
)
from vyper.venom.context import IRFunction
from vyper.venom.effects import Effects

NONIDEMPOTENT_INSTRUCTIONS = frozenset(["log", "call", "staticcall", "delegatecall", "invoke"])

# instructions that queries info about current
# environment this is done because we know that
# all these instruction should have always
# the same value in function


# instruction that dont need to be stored in available expression
UNINTERESTING_OPCODES = frozenset(
    [
        "calldatasize",
        "gaslimit",
        "address",
        "codesize",
        "store",
        "phi",
        "param",
        "nop",
        "returndatasize",
        "gas",
        "gasprice",
        "origin",
        "coinbase",
        "timestamp",
        "number",
        "prevrandao",
        "chainid",
        "basefee",
        "blobbasefee",
        "pc",
        "msize",
    ]
)

def get_read_effects(opcode, ignore_msize):
    effs = effects.reads.get(opcode, effects.EMPTY)

    if ignore_msize:
        effs &= ~effects.MSIZE

    return effs

def get_write_effects(opcode, ignore_msize):
    effs = effects.writes.get(opcode, effects.EMPTY)

    if ignore_msize:
        effs &= ~effects.MSIZE

    return effs


@dataclass(frozen=True)
# expression is like the nested representation of an instruction.
# note that Expression can include no-output instructions
class _Expression:
    opcode: str
    # the child is either expression of operand since
    # there are possibilities for cycles
    operands: list[IROperand | _Expression]

    effect_generation: tuple[Optional[effects.Effect], ...]

    # pointer to the basic block that this Expression is valid in
    bb: IRBasicBlock

    @classmethod
    def create(cls, opcode, operands, generation, bb, ignore_msize):
        read_effects = get_read_effects(opcode, ignore_msize)

        # we represent the generation with `None` for the effects
        # that are not read by this Expression.
        assert len(generation) == len(effects.Effects), (opcode, generation)
        read_generation = tuple()
        for g_i, eff in zip(generation, effects.Effects):
            t = g_i
            if not (read_effects & eff):
                t = None
            read_generation += t,

        return cls(opcode, operands, read_generation, bb)

    def with_fresh_generation(self, bb: IRBasicBlock):
        fresh_generation = tuple(0 if g_i else None for g_i in self.effect_generation)
        return self.__class__(self.opcode, self.operands, fresh_generation, bb)

    # equality for lattices only based on original instruction
    def __eq__(self, other) -> bool:
        if not isinstance(other, _Expression):
            return False
        return self.same(other)

    @cached_property
    # Unfortunately the hash has been the performance
    # bottle neck in some cases so I cached the value
    def _cached_hash(self):
        # the reason for the sort is that some opcodes could
        # be commutative and in that case the order of the
        # operands would not matter (so this is needed)
        # for correct implementation of hash (x == x => hash(x) == hash(y))
        return hash((self.opcode, tuple(sorted(hash(op) for op in self.operands))))

    def __hash__(self) -> int:
        return self._cached_hash

    # Full equality for expressions based on opcode and operands
    def same(self, other) -> bool:
        return same(self, other)

    def __repr__(self) -> str:
        if self.opcode == "store":
            assert len(self.operands) == 1, "wrong store"
            return repr(self.operands[0])
        res = self.opcode + "("
        res += ",".join(repr(op) for op in self.operands)
        res += ")"
        return res

    @cached_property
    def depth(self) -> int:
        max_depth = 0
        for op in self.operands:
            if isinstance(op, _Expression):
                d = op.depth
                if d > max_depth:
                    max_depth = d
        return max_depth + 1

    def get_reads(self, ignore_msize) -> Effects:
        return get_read_effects(self.opcode, ignore_msize)

    def get_writes(self, ignore_msize) -> Effects:
        return get_write_effects(self.opcode, ignore_msize)

    @property
    def is_commutative(self) -> bool:
        return self.opcode in COMMUTATIVE_INSTRUCTIONS


def same(a: IROperand | _Expression, b: IROperand | _Expression) -> bool:
    if isinstance(a, IROperand) and isinstance(b, IROperand):
        return a.value == b.value

    if not isinstance(a, _Expression) or not isinstance(b, _Expression):
        return False

    if a is b:
        return True

    if a.opcode != b.opcode:
        return False

    # due to lattice_meet, we never call same() on Expressions generated in
    # different basic blocks
    assert a.bb == b.bb

    if a.effect_generation != b.effect_generation:
        return False

    # Early return special case for commutative instructions
    if a.is_commutative:
        if same(a.operands[0], b.operands[1]) and same(a.operands[1], b.operands[0]):
            return True

    # General case
    for self_op, other_op in zip(a.operands, b.operands):
        if type(self_op) is not type(other_op):
            return False
        if isinstance(self_op, IROperand) and self_op != other_op:
            return False
        if isinstance(self_op, _Expression) and self_op is not other_op:
            return False

    return True


class _AvailableExpression:
    """
    Class that holds available expression
    and provides API for handling them
    """

    exprs: dict[_Expression, list[IRInstruction]]

    def __init__(self):
        self.exprs = dict()

    def __eq__(self, other) -> bool:
        if not isinstance(other, _AvailableExpression):
            return False

        return self.exprs == other.exprs

    def __repr__(self) -> str:
        res = "available expr\n"
        for key, val in self.exprs.items():
            res += f"\t{key}: {val}\n"
        return res

    def add(self, expr: _Expression, src_inst: IRInstruction):
        if expr not in self.exprs:
            self.exprs[expr] = []
        self.exprs[expr].append(src_inst)

    def get_source_instruction(self, expr: _Expression) -> IRInstruction | None:
        """
        Get source instruction of expression if currently available
        """
        tmp = self.exprs.get(expr)
        if tmp is not None:
            # arbitrarily choose the first instruction
            return tmp[0]
        return None

    def output_expressions(self, generation) -> _AvailableExpression:
        res = _AvailableExpression()

        for k, v in self.exprs.items():
            if not compatible_generation(k.effect_generation, generation):
                continue
            res.exprs[k] = v

        return res

    def copy(self) -> _AvailableExpression:
        res = _AvailableExpression()
        for k, v in self.exprs.items():
            res.exprs[k] = v.copy()
        return res

    @staticmethod
    def lattice_meet(bb: IRBasicBlock, lattices: list[_AvailableExpression]):
        if len(lattices) == 0:
            return _AvailableExpression()

        # construct starting lattice
        res = _AvailableExpression()
        for expr, v in lattices[0].exprs.items():
            new_expr = expr.with_fresh_generation(bb)
            res.exprs[new_expr] = v.copy()

        for lattice in lattices[1:]:
            for expr, insts in lattice.exprs.items():
                expr = expr.with_fresh_generation(bb)

                if expr not in res.exprs:
                    continue

                new_insts = _list_intersection(insts, res.exprs[expr])

                # no common instruction which we can use
                # (TODO: insert a phi instruction instead)
                if len(new_insts) == 0:
                    del res.exprs[expr]

        return res

def _list_intersection(xs: list, ys: list):
    return [x for x in xs if x in ys]

NULL_GENERATION = tuple(0 for _ in effects.Effects)

def _bump_generation(generation: tuple[int, ...], write_effect: effects.Effects):

    # TODO: make this readable
    assert len(generation) == len(effects.Effects)
    return tuple(g + (1 if (write_effect & eff) else 0) for (g, eff) in zip(generation, effects.Effects))


def compatible_generation(generation: tuple[int, ...], to_compare: tuple[Optional[int], ...]):
    for lhs, rhs in zip(generation, to_compare):
        if lhs != rhs and rhs is not None:
            return False

    return True


class CSEAnalysis(IRAnalysis):
    inst_to_expr: dict[IRInstruction, _Expression]
    dfg: DFGAnalysis

    # cache entry available expression for bb
    bb_ins: dict[IRBasicBlock, _AvailableExpression]

    # for merging at lattice join point
    bb_outs: dict[IRBasicBlock, _AvailableExpression]

    ignore_msize: bool

    def __init__(self, analyses_cache: IRAnalysesCache, function: IRFunction):
        super().__init__(analyses_cache, function)
        self.analyses_cache.request_analysis(CFGAnalysis)
        dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        assert isinstance(dfg, DFGAnalysis)
        self.dfg = dfg

        self.inst_to_expr = dict()
        self.inst_to_generation = dict()
        self.bb_ins = dict()
        self.bb_outs = dict()

        self.ignore_msize = not self._contains_msize()

    #@profileit()
    def analyze(self):
        worklist = deque()
        worklist.append(self.function.entry)
        while len(worklist) > 0:
            bb: IRBasicBlock = worklist.popleft()
            if self._handle_bb(bb):
                worklist.extend(bb.cfg_out)

    def get_from_same_bb(self, inst: IRInstruction, expr: _Expression) -> list[IRInstruction]:
        available_exprs = self.bb_ins[inst.parent]
        res = available_exprs.exprs[expr]
        return [i for i in res if i != inst and i.parent == inst.parent]

    # msize effect should be only necessery
    # to be handled when there is a possibility
    # of msize read otherwise it should not make difference
    # for this analysis
    def _contains_msize(self) -> bool:
        for bb in self.function.get_basic_blocks():
            for inst in bb.instructions:
                if inst.opcode == "msize":
                    return True
        return False

    def copy1(self, x):
        # profiler trick, TODO remove me
        return x.copy()

    def copy2(self, x):
        return x.copy()

    def _handle_bb(self, bb: IRBasicBlock) -> bool:
        available_exprs = _AvailableExpression.lattice_meet(
            bb,
            [self.bb_outs.get(pred, _AvailableExpression()) for pred in bb.cfg_in],
        )

        if bb in self.bb_ins and self.bb_ins[bb] == available_exprs:
            # no change
            return False

        self.bb_ins[bb] = available_exprs

        change = False

        generation = NULL_GENERATION

        for inst in bb.instructions:
            self.inst_to_generation[inst] = generation

            if inst.opcode in UNINTERESTING_OPCODES or inst.opcode in BB_TERMINATORS:
                continue

            expr = self._get_expression(inst, generation, available_exprs)
            if inst == available_exprs.get_source_instruction(expr):
                # bump the generation, unless the instruction is substitutable
                generation = _bump_generation(generation, get_write_effects(inst.opcode, ignore_msize))

            # nonidempotent instruction effect other instructions
            # but since it cannot be substituted it does not have
            # to be added to available exprs
            if inst.opcode in NONIDEMPOTENT_INSTRUCTIONS:
                continue

            # read effects do not overlap write effects
            expr_effects = expr.get_writes(self.ignore_msize) & expr.get_reads(self.ignore_msize)
            if expr_effects == effects.EMPTY:
                available_exprs.add(expr, inst)

        if bb not in self.bb_outs or available_exprs != self.bb_outs[bb]:
            self.bb_outs[bb] = available_exprs.output_expressions(generation)
            # change is only necessery when the output of the
            # basic block is changed (otherwise it wont affect rest)
            change |= True

        return change

    # promote operand to expr if we can
    def _get_operand(
        self, op: IROperand, generation: tuple, available_exprs: _AvailableExpression
    ) -> IROperand | _Expression:
        if not isinstance(op, IRVariable):
            return op
        inst = self.dfg.get_producing_instruction(op)
        assert inst is not None, op
        # phis can create dataflow loops (which would be infinite depth
        # expressions) so we ignore them for now.
        if inst.opcode == "phi":
            return op

        if inst.opcode == "store":
            return self._get_operand(inst.operands[0], generation, available_exprs)

        if inst in self.inst_to_expr:
            # TODO:
            # e = self.inst_to_expr[inst]
            # src = available_exprs.get_source_instruction(expr)
            # if src is None:
            #     # return the original instruction
            #     return e
            # return self.inst_to_expr[src]

            e = self.inst_to_expr[inst]
            same_insts = available_exprs.exprs.get(e, [])
            if inst in same_insts:
                return self.inst_to_expr[same_insts[0]]
            return e

        assert inst.opcode in UNINTERESTING_OPCODES
        return self._get_expression(inst, generation, available_exprs)

    # todo: rename to `get_common_subexpression`?
    def get_expression(self, inst: IRInstruction) -> tuple[_Expression, IRInstruction]:
        available_exprs = self.bb_ins[inst.parent]
        generation = self.inst_to_generation[inst]

        expr = self.inst_to_expr.get(inst)
        if expr is None:
            # expr = self._get_operand(inst.output)
            expr = self._get_expression(inst, generation, available_exprs)

        src = available_exprs.get_source_instruction(expr)
        if src is None:
            # return the original instruction
            src = inst

        return (expr, src)

    def _get_expression(
        self, inst: IRInstruction, generation: tuple, available_exprs: _AvailableExpression
    ) -> _Expression:
        # create expression
        operands: list[IROperand | _Expression] = [
            self._get_operand(op, generation, available_exprs) for op in inst.operands
        ]
        expr = _Expression.create(inst.opcode, operands, generation, inst.parent, self.ignore_msize)

        src_inst = available_exprs.get_source_instruction(expr)
        if src_inst is not None:
            same_expr = self.inst_to_expr[src_inst]
            if same_expr is not None:
                self.inst_to_expr[inst] = same_expr
                return same_expr

        self.inst_to_expr[inst] = expr
        return expr
