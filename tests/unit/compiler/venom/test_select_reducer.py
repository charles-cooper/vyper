import pytest

from tests.venom_utils import PrePostChecker
from vyper.venom.passes import RemoveUnusedVariablesPass, SelectReducer, SimplifyCFGPass

pytestmark = pytest.mark.hevm

_check_pre_post = PrePostChecker([SelectReducer, RemoveUnusedVariablesPass, SimplifyCFGPass])


def test_simple_select_reducer():
    pre = """
    main:
        %p1 = param
        %p2 = param
        %cond = param

        jnz %cond, @br1, @br2
    br1:
        %r1 = %p1
        jmp @join
    br2:
        %r2 = %p2
        jmp @join
    join:
        %q = phi @br1, %r1, @br2, %r2
        sink %q
    """

    post = """
    main:
        %p1 = param
        %p2 = param
        %cond = param
        %1 = iszero %cond
        %2 = xor %p2, %p1
        %3 = mul %2, %1
        %4 = xor %p1, %3
        %q = %4
        sink %q
    """
    _check_pre_post(pre, post)
