import pytest

from vyper.utils import evm_not
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.basicblock import IRLiteral
from vyper.venom.context import IRContext
from vyper.venom.passes import ReduceLiteralsCodesize

should_invert = [2**256 - 1] + [((2**i) - 1) << (256 - i) for i in range(121, 256 + 1)]


@pytest.mark.parametrize("orig_value", should_invert)
def test_literal_codesize_ff_inversion(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")
    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "not"
    assert evm_not(bb.instructions[0].operands[0].value) == orig_value
    assert bb.instructions[1].opcode == "stop"


should_not_invert = [
    # goes from one since the first one can be optimized with
    # combination of two optimizations
    ((2**255 - 1) >> i) << i
    for i in range(1, 3 * 8)
]


@pytest.mark.parametrize("orig_value", should_not_invert)
def test_literal_codesize_no_inversion(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")
    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "store"
    assert bb.instructions[0].operands[0].value == orig_value
    assert bb.instructions[1].opcode == "stop"


should_shl = (
    [2**i for i in range(3 * 8, 255)]
    + [((2**i) - 1) << (256 - i) for i in range(1, 121)]
    + [((2**255 - 1) >> i) << i for i in range(3 * 8, 254)]
)


@pytest.mark.parametrize("orig_value", should_shl)
def test_literal_codesize_shl(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")
    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "shl"
    op0, op1 = bb.instructions[0].operands
    assert op0.value << op1.value == orig_value
    assert bb.instructions[1].opcode == "stop"


should_not_shl = [1 << i for i in range(0, 3 * 8)] + [
    0x0,
    (((2 ** (256 - 2)) - 1) << (2 * 8)) ^ (2**255),
]


@pytest.mark.parametrize("orig_value", should_not_shl)
def test_literal_codesize_no_shl(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")
    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "store"
    assert bb.instructions[0].operands[0].value == orig_value
    assert bb.instructions[1].opcode == "stop"


should_do_both = [(2**256 - 1) ^ (1 << i) for i in range(4 * 8, 256)]


@pytest.mark.parametrize("orig_value", should_do_both)
def test_literal_codesize_both(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")

    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "shl"
    op0, op1 = bb.instructions[0].operands
    assert evm_not(op0.value << op1.value) == orig_value
    assert bb.instructions[1].opcode == "not"
    assert bb.instructions[1].operands[0] == bb.instructions[0].output
    assert bb.instructions[2].opcode == "stop"


should_not_do_any = [(2**256 - 1) ^ (1 << i) ^ 1 ^ 2**255 for i in range(4 * 8, 255)]


@pytest.mark.parametrize("orig_value", should_not_do_any)
def test_literal_codesize_nothing(orig_value):
    ctx = IRContext()
    fn = ctx.create_function("_global")
    bb = fn.get_basic_block()

    bb.append_instruction("store", IRLiteral(orig_value))
    bb.append_instruction("stop")

    ac = IRAnalysesCache(fn)
    ReduceLiteralsCodesize(ac, fn).run_pass()

    assert bb.instructions[0].opcode == "store"
    assert bb.instructions[0].operands[0].value == orig_value
    assert bb.instructions[1].opcode == "stop"
