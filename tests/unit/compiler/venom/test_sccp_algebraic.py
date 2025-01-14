import pytest

from tests.venom_utils import assert_ctx_eq, parse_from_basic_block
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.passes import (
    SCCP,
    AlgebraicOptimizationPass,
    RemoveUnusedVariablesPass,
    StoreElimination,
)

"""
Test abstract binop+unop optimizations in sccp and algebraic optimizations pass
"""


def _sccp_algebraic_runner(pre, post):
    ctx = parse_from_basic_block(pre)

    for fn in ctx.functions.values():
        ac = IRAnalysesCache(fn)
        StoreElimination(ac, fn).run_pass()
        SCCP(ac, fn).run_pass()
        AlgebraicOptimizationPass(ac, fn).run_pass()
        SCCP(ac, fn).run_pass()
        StoreElimination(ac, fn).run_pass()
        RemoveUnusedVariablesPass(ac, fn).run_pass()

    assert_ctx_eq(ctx, parse_from_basic_block(post))


def test_sccp_algebraic_opt_sub_xor():
    # x - x -> 0
    # x ^ x -> 0
    pre = """
    _global:
        %par = param
        %1 = sub %par, %par
        %2 = xor %par, %par
        return %1, %2
    """
    post = """
    _global:
        %par = param
        return 0, 0
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_zero_sub_xor():
    # x + 0 == x - 0 == x ^ 0 -> x
    # this cannot be done for 0 - x
    pre = """
    _global:
        %par = param
        %1 = sub %par, 0
        %2 = xor %par, 0
        %3 = add 0, %par
        %4 = sub 0, %par
        %5 = sub -1, %par
        return %1, %2, %3, %4, %5
    """
    post = """
    _global:
        %par = param
        %4 = sub 0, %par
        %5 = not %par
        return %par, %par, %par, %4, %5
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_xor_max():
    # x ^ 0xFF..FF -> not x
    max_uint256 = (2**256) - 1
    pre = f"""
    _global:
        %par = param
        %tmp = {max_uint256}
        %1 = xor %tmp, %par
        return %1
    """
    post = """
    _global:
        %par = param
        %1 = not %par
        return %1
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_shift():
    # x << 0 == x >> 0 == x (sar) 0 -> x
    # sar is right arithmetic shift
    pre = """
    _global:
        %par = param
        %1 = shl 0, %par
        %2 = shr 0, %1
        %3 = sar 0, %2
        return %1, %2, %3
    """
    post = """
    _global:
        %par = param
        return %par, %par, %par
    """

    _sccp_algebraic_runner(pre, post)


@pytest.mark.parametrize("opcode", ("mul", "and", "div", "sdiv", "mod", "smod"))
def test_mul_by_zero(opcode):
    # x * 0 == 0 * x == x % 0 == 0 % x == x // 0 == 0 // x == x & 0 == 0 & x -> 0
    pre = f"""
    _global:
        %par = param
        %1 = {opcode} 0, %par
        %2 = {opcode} %par, 0
        return %1, %2
    """
    post = """
    _global:
        %par = param
        return 0, 0
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_multi_neutral_elem():
    # x * 1 == 1 * x == x / 1 -> x
    # checks for non comutative ops
    pre = """
    _global:
        %par = param
        %1_1 = mul 1, %par
        %1_2 = mul %par, 1
        %2_1 = div 1, %par
        %2_2 = div %par, 1
        %3_1 = sdiv 1, %par
        %3_2 = sdiv %par, 1
        return %1_1, %1_2, %2_1, %2_2, %3_1, %3_2
    """
    post = """
    _global:
        %par = param
        %2_1 = div 1, %par
        %3_1 = sdiv 1, %par
        return %par, %par, %2_1, %par, %3_1, %par
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_mod_zero():
    # x % 1 -> 0
    pre = """
    _global:
        %par = param
        %1 = mod %par, 1
        %2 = smod %par, 1
        return %1, %2
    """
    post = """
    _global:
        %par = param
        return 0, 0
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_and_max():
    # x & 0xFF..FF == 0xFF..FF & x -> x
    max_uint256 = 2**256 - 1
    pre = f"""
    _global:
        %par = param
        %tmp = {max_uint256}
        %1 = and %par, %tmp
        %2 = and %tmp, %par
        return %1, %2
    """
    post = """
    _global:
        %par = param
        return %par, %par
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_mul_div_to_shifts():
    # x * 2**n -> x << n
    # x / 2**n -> x >> n
    pre = """
    _global:
        %par = param
        %1 = mod %par, 8
        %2 = mul %par, 16
        %3 = div %par, 4
        return %1, %2, %3
    """
    post = """
    _global:
        %par = param
        %1 = and 7, %par
        %2 = shl 4, %par
        %3 = shr 2, %par
        return %1, %2, %3
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_exp():
    # x ** 0 == 0 ** x -> 1
    # x ** 1 -> x
    pre = """
    _global:
        %par = param
        %1 = exp %par, 0
        %2 = exp 1, %par
        %3 = exp 0, %par
        %4 = exp %par, 1
        return %1, %2, %3, %4
    """
    post = """
    _global:
        %par = param
        %3 = iszero %par
        return 1, 1, %3, %par
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_compare_self():
    # x < x == x > x -> 0
    pre = """
    _global:
        %par = param
        %tmp = %par
        %1 = gt %tmp, %par
        %2 = sgt %tmp, %par
        %3 = lt %tmp, %par
        %4 = slt %tmp, %par
        return %1, %2, %3, %4
    """
    post = """
    _global:
        %par = param
        return 0, 0, 0, 0
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_or_eq():
    # x | 0 -> x
    # x | 0xFF..FF -> 0xFF..FF
    # (x == 0) == (0 == x) -> iszero x
    # x == x -> 1
    max_uint256 = 2**256 - 1
    pre = f"""
    _global:
        %par = param
        %1 = or %par, 0
        %2 = or %par, {max_uint256}
        %3 = eq %par, 0
        %4 = eq 0, %par
        %5 = eq %par, %par
        %6 = eq %par, {max_uint256}
        return %1, %2, %3, %4, %5, %6
    """
    post = f"""
    _global:
        %par = param
        %3 = iszero %par
        %4 = iszero %par
        %8 = not %par
        %6 = iszero %8
        return %par, {max_uint256}, %3, %4, 1, %6
    """

    _sccp_algebraic_runner(pre, post)


def test_sccp_algebraic_opt_boolean_or_eq():
    # x == 1 -> iszero (x xor 1) if it is only used as boolean
    # x | (non zero) -> 1 if it is only used as boolean
    pre = """
    _global:
        %par = param
        %1 = eq %par, 1
        %2 = eq %par, 1
        assert %1
        %3 = or %par, 123
        %4 = or %par, 123
        assert %3
        return %2, %4
    """
    post = """
    _global:
        %par = param
        %5 = xor 1, %par
        %1 = iszero %5
        %2 = eq 1, %par
        assert %1
        %4 = or 123, %par
        nop
        return %2, %4
    """

    _sccp_algebraic_runner(pre, post)


def test_compare_never():
    # unsigned x > 0xFF..FF == x < 0 -> 0
    # signed: x > MAX_SIGNED (0x3F..FF) == x < MIN_SIGNED (0xF0..00) -> 0
    min_int256 = -(2**255)
    max_int256 = 2**255 - 1
    min_uint256 = 0
    max_uint256 = 2**256 - 1
    pre = f"""
    _global:
        %par = param

        %1 = slt %par, {min_int256}
        %2 = sgt %par, {max_int256}
        %3 = lt %par, {min_uint256}
        %4 = gt %par, {max_uint256}

        return %1, %2, %3, %4
    """
    post = """
    _global:
        %par = param
        return 0, 0, 0, 0
    """

    _sccp_algebraic_runner(pre, post)


def test_comparison_zero():
    # x > 0 => iszero(iszero x)
    # 0 < x => iszero(iszero x)
    pre = """
    _global:
        %par = param
        %1 = lt 0, %par
        %2 = gt %par, 0
        return %1, %2
    """
    post = """
    _global:
        %par = param
        %3 = iszero %par
        %1 = iszero %3
        %4 = iszero %par
        %2 = iszero %4
        return %1, %2
    """

    _sccp_algebraic_runner(pre, post)


def test_comparison_almost_never():
    # unsigned:
    #   x < 1 => eq x 0 => iszero x
    #   MAX_UINT - 1 < x => eq x MAX_UINT => iszero(not x)
    # signed
    #   x < MIN_INT + 1 => eq x MIN_INT
    #   MAX_INT - 1 < x => eq x MAX_INT

    max_uint256 = 2**256 - 1
    max_int256 = 2**255 - 1
    min_int256 = -(2**255)
    pre = f"""
    _global:
        %par = param
        %1 = lt %par, 1
        %2 = gt 1, %par
        %3 = gt %par, {max_uint256 - 1}
        %4 = sgt %par, {max_int256 - 1}
        %5 = slt %par, {min_int256 + 1}
        return %1, %2, %3, %4, %5
    """
    post = f"""
    _global:
        %par = param
        ; first into eq 0 %par then into iszere
        %1 = iszero %par
        %2 = iszero %par
        ; this also goes through eq
        %6 = not %par
        %3 = iszero %6
        %4 = eq {max_int256}, %par
        %5 = eq {min_int256}, %par
        return %1, %2, %3, %4, %5
    """

    _sccp_algebraic_runner(pre, post)


def test_comparison_almost_always():
    # unsigned
    #   x > 0 => iszero(iszero x)
    #   0 < x => iszero(iszero x)
    #   x < MAX_UINT => iszero(eq x MAX_UINT) => iszero(iszero(not x))
    # signed
    #   x < MAX_INT => iszero(eq MAX_INT) => iszero(iszero(xor x))

    max_uint256 = 2**256 - 1
    max_int256 = 2**255 - 1
    min_int256 = -(2**255)

    pre = f"""
    _global:
        %par = param
        %1 = lt 0, %par
        %2 = gt %par, 0
        %3 = lt %par, {max_uint256}
        assert %3
        %4 = slt %par, {max_int256}
        assert %4
        %4 = sgt %par, {min_int256}
        assert %4
        return %1, %2
    """
    post = f"""
    _global:
        %par = param
        %5 = iszero %par
        %1 = iszero %5
        %6 = iszero %par
        %2 = iszero %6
        %10 = not %par
        %7 = iszero %10
        %3 = iszero %7
        assert %3
        %11 = xor %par, {max_int256}
        %8 = iszero %11
        %4 = iszero %8
        assert %4
        %12 = xor %par, {min_int256}
        %9 = iszero %12
        %4 = iszero %9
        assert %4
        return %1, %2
    """

    _sccp_algebraic_runner(pre, post)


@pytest.mark.parametrize("val", (100, 2, 3, -100))
def test_comparison_ge_le(val):
    # iszero(x < 100) => 99 <= x
    # iszero(x > 100) => 101 >= x

    up = val + 1
    down = val - 1

    abs_val = abs(val)
    abs_up = abs_val + 1
    abs_down = abs_val - 1

    pre = f"""
    _global:
        %par = param
        %1 = lt %par, {abs_val}
        %3 = gt %par, {abs_val}
        %4 = iszero %3
        %2 = iszero %1
        %5 = slt %par, {val}
        %7 = sgt %par, {val}
        %6 = iszero %5
        %8 = iszero %7
        return %2, %4, %6, %8
    """
    post = f"""
    _global:
        %par = param
        %1 = lt {abs_down}, %par
        %3 = gt {abs_up}, %par
        %5 = slt {down}, %par
        %7 = sgt {up}, %par
        return %1, %3, %5, %7
    """

    _sccp_algebraic_runner(pre, post)
