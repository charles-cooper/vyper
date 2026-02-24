from __future__ import annotations

from dataclasses import dataclass

from vyper.venom.analysis import DFGAnalysis
from vyper.venom.basicblock import IRInstruction, IRLabel, IROperand, IRVariable
from vyper.venom.function import IRFunction
from vyper.venom.memory_location import memory_write_ops
from vyper.venom.passes.base_pass import IRGlobalPass


@dataclass(frozen=True)
class _FnParamInfo:
    # params that are passed via invoke operands (all params except return_pc)
    invoke_params: tuple[IRVariable, ...]
    # map param var -> invoke operand index
    invoke_param_index: dict[IRVariable, int]


class ReadonlyMemoryArgsAnalysisPass(IRGlobalPass):
    """
    Infer readonly invoke-arg positions from Venom IR.

    The result is stored on each IRFunction as:
      `_readonly_memory_invoke_arg_idxs: tuple[int, ...]`
    where indices are relative to invoke stack args (excluding label).

    Analysis is conservative and interprocedural:
    - local writes through a parameter-derived pointer mark it mutable
    - passing a parameter-derived pointer to a non-readonly callee arg marks it mutable
    - fixed-point iteration propagates mutability through the call graph
    """

    def run_pass(self):
        infos = {fn: self._collect_param_info(fn) for fn in self.ctx.get_functions()}

        readonly_by_fn: dict[IRFunction, tuple[bool, ...]] = {}
        for fn, info in infos.items():
            readonly_by_fn[fn] = tuple(True for _ in range(len(info.invoke_params)))

        changed = True
        while changed:
            changed = False
            for fn, info in infos.items():
                new_state = self._analyze_fn(fn, info, readonly_by_fn)
                if new_state != readonly_by_fn[fn]:
                    readonly_by_fn[fn] = new_state
                    changed = True

        for fn, state in readonly_by_fn.items():
            idxs = tuple(i for i, is_ro in enumerate(state) if is_ro)
            fn._readonly_memory_invoke_arg_idxs = idxs

    def _collect_param_info(self, fn: IRFunction) -> _FnParamInfo:
        # Internal-call convention: last param is return_pc.
        params = [inst.output for inst in fn.entry.param_instructions]
        if len(params) == 0:
            return _FnParamInfo(tuple(), {})

        invoke_params = tuple(params[:-1])
        invoke_param_index = {var: i for i, var in enumerate(invoke_params)}
        return _FnParamInfo(invoke_params, invoke_param_index)

    def _analyze_fn(
        self, fn: IRFunction, info: _FnParamInfo, readonly_by_fn: dict[IRFunction, tuple[bool, ...]]
    ) -> tuple[bool, ...]:
        n = len(info.invoke_params)
        if n == 0:
            return ()

        mutable = [False] * n
        dfg = self.analyses_caches[fn].request_analysis(DFGAnalysis)

        root_memo: dict[IRVariable, int | None] = {}
        root_active: set[IRVariable] = set()

        def root_param_index(op: IROperand) -> int | None:
            if not isinstance(op, IRVariable):
                return None
            return root_param_index_var(op)

        def root_param_index_var(var: IRVariable) -> int | None:
            if var in root_memo:
                return root_memo[var]
            if var in root_active:
                return None

            idx = info.invoke_param_index.get(var, None)
            if idx is not None:
                root_memo[var] = idx
                return idx

            root_active.add(var)
            inst = dfg.get_producing_instruction(var)
            idx = self._root_from_inst(inst, root_param_index_var)
            root_active.remove(var)
            root_memo[var] = idx
            return idx

        for bb in fn.get_basic_blocks():
            for inst in bb.instructions:
                if inst.opcode == "invoke":
                    self._handle_invoke(inst, mutable, root_param_index, readonly_by_fn)
                    continue

                write_ofst = memory_write_ops(inst).ofst
                if write_ofst is None:
                    continue

                idx = root_param_index(write_ofst)
                if idx is not None:
                    mutable[idx] = True

        return tuple(not is_mut for is_mut in mutable)

    def _handle_invoke(
        self,
        inst: IRInstruction,
        mutable: list[bool],
        root_param_index,
        readonly_by_fn: dict[IRFunction, tuple[bool, ...]],
    ) -> None:
        target = inst.operands[0]
        if not isinstance(target, IRLabel):
            return

        callee = self.ctx.functions.get(target, None)

        for op_idx, op in enumerate(inst.operands[1:], start=1):
            caller_idx = root_param_index(op)
            if caller_idx is None:
                continue

            callee_arg_idx = op_idx - 1
            if callee is None:
                mutable[caller_idx] = True
                continue

            callee_state = readonly_by_fn.get(callee, ())
            if callee_arg_idx >= len(callee_state) or not callee_state[callee_arg_idx]:
                mutable[caller_idx] = True

    def _root_from_inst(self, inst: IRInstruction | None, root_param_index_var) -> int | None:
        if inst is None:
            return None

        op = inst.opcode
        if op == "assign":
            src = inst.operands[0]
            if isinstance(src, IRVariable):
                return root_param_index_var(src)
            return None

        if op == "gep":
            return self._root_from_gep(inst, root_param_index_var)

        if op == "add":
            return self._root_from_add(inst, root_param_index_var)

        if op == "sub":
            return self._root_from_sub(inst, root_param_index_var)

        if op == "phi":
            roots = set()
            for _, v in inst.phi_operands:
                r = root_param_index_var(v)
                if r is None:
                    return None
                roots.add(r)
            if len(roots) == 1:
                return next(iter(roots))
            return None

        if op == "select":
            # select cond, a, b  (stored as [b, a, cond])
            a = inst.operands[1]
            b = inst.operands[2]
            if not isinstance(a, IRVariable) or not isinstance(b, IRVariable):
                return None
            ra = root_param_index_var(a)
            rb = root_param_index_var(b)
            if ra is not None and ra == rb:
                return ra
            return None

        return None

    def _root_from_add(self, inst: IRInstruction, root_param_index_var) -> int | None:
        roots = set()
        for op in inst.operands:
            if not isinstance(op, IRVariable):
                continue
            r = root_param_index_var(op)
            if r is not None:
                roots.add(r)

        if len(roots) == 1:
            return next(iter(roots))
        return None

    def _root_from_sub(self, inst: IRInstruction, root_param_index_var) -> int | None:
        # IR order for sub(a, b) is [b, a].
        if len(inst.operands) != 2:
            return None
        b, a = inst.operands
        if not isinstance(a, IRVariable):
            return None

        ra = root_param_index_var(a)
        if ra is None:
            return None

        rb = root_param_index_var(b) if isinstance(b, IRVariable) else None
        if rb is None or rb == ra:
            return ra
        return None

    def _root_from_gep(self, inst: IRInstruction, root_param_index_var) -> int | None:
        # IR order for gep(ptr, offset) is [ptr, offset].
        if len(inst.operands) != 2:
            return None
        base, offset = inst.operands
        if not isinstance(base, IRVariable):
            return None

        rbase = root_param_index_var(base)
        if rbase is None:
            return None

        # If offset is derived from a different parameter root, we can't
        # attribute the resulting pointer to a single param.
        roffset = root_param_index_var(offset) if isinstance(offset, IRVariable) else None
        if roffset is None or roffset == rbase:
            return rbase
        return None
