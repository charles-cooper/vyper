import pytest

from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.check_venom import check_venom_ctx
from vyper.venom.parser import parse_venom
from vyper.venom.passes import LoopUnrollingPass

pytestmark = pytest.mark.hevm


def _run_unroll(source: str, **kwargs):
    ctx = parse_venom(source)
    fn = next(iter(ctx.functions.values()))
    ac = IRAnalysesCache(fn)
    LoopUnrollingPass(ac, fn).run_pass(**kwargs)
    check_venom_ctx(ctx)
    return fn


def test_unroll_ssa_exact_trip_count():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 3
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels
    assert any("unroll_body" in label for label in labels)


def test_unroll_ssa_eq_condition_with_header_bound_and_assign_chain():
    src = """
    function main {
    entry:
        %start = 0
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %i_alias = %i
        %end_h = 3
        %done = eq %i_alias, %end_h
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %start
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels
    assert any("unroll_body" in label for label in labels)


def test_unroll_ssa_zero_trip_count_removes_loop():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 0
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 7
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels

    entry = fn.get_basic_block("entry")
    assert entry.last_instruction.opcode == "jmp"
    assert entry.last_instruction.operands[0].value == "exit"


def test_skip_ssa_when_trip_count_exceeds_limit():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 9
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src, max_full_unroll_iterations=8)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" in labels
    assert "body" in labels
    assert not any("unroll_body" in label for label in labels)


def test_skip_ssa_when_phi_value_escapes_loop():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 2
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %i
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" in labels
    assert "body" in labels
    assert not any("unroll_body" in label for label in labels)


def test_unroll_ssa_phi_loop_with_guarded_upper_bound():
    src = """
    function main {
    entry:
        %start = 0
        %trip = source
        %ok_bound = lt 3, %trip
        %bound_check = iszero %ok_bound
        assert %bound_check
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 7
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels
    assert any("unroll_cond_body" in label for label in labels)
    assert any("unroll_body" in label for label in labels)


def test_non_ssa_loop_is_not_unrolled():
    src = """
    function main {
    entry:
        %i = 0
        %end = 3
        jmp @cond
    cond:
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        jmp @incr
    incr:
        %i = add %i, 1
        jmp @cond
    exit:
        sink %i
    }
    """

    fn = _run_unroll(src)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "cond" in labels
    assert "body" in labels
    assert "incr" in labels
    assert not any("unroll_" in label for label in labels)


def test_size_reduction_mode_skips_unprofitable_exact_unroll():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 3
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src, require_size_reduction=True)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" in labels
    assert "body" in labels
    assert not any("unroll_body" in label for label in labels)


def test_size_reduction_mode_allows_profitable_exact_unroll():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 1
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 42
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src, require_size_reduction=True)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels
    assert any("unroll_body" in label for label in labels)


def test_size_reduction_mode_skips_guarded_unroll():
    src = """
    function main {
    entry:
        %start = 0
        %trip = source
        %ok_bound = lt 3, %trip
        %bound_check = iszero %ok_bound
        assert %bound_check
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %tmp = add %i, 7
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src, require_size_reduction=True)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" in labels
    assert "body" in labels
    assert not any("unroll_" in label for label in labels)


def test_size_reduction_mode_allows_elision_aware_unroll():
    src = """
    function main {
    entry:
        %start = 0
        %trip = 2
        %end = add %start, %trip
        jmp @header
    header:
        %i = phi @entry, %start, @body, %next
        %cmp = xor %i, %end
        %done = iszero %cmp
        jnz %done, @exit, @body
    body:
        %dst = 64
        %src = 32
        %len = 32
        mcopy %dst, %src, %len
        %next = add %i, 1
        jmp @header
    exit:
        sink %trip
    }
    """

    fn = _run_unroll(src, require_size_reduction=True, consider_elision_benefit=True)
    labels = [bb.label.value for bb in fn.get_basic_blocks()]
    assert "header" not in labels
    assert "body" not in labels
    assert any("unroll_body" in label for label in labels)
