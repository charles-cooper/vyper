from vyper.compiler import compile_code
from vyper.compiler.settings import OptimizationLevel, Settings, VenomOptimizationFlags


def test_annassign_internal_call_result_reuses_return_buffer():
    code = """
@internal
def _id(a: uint256[2]) -> uint256[2]:
    return a

@internal
def _sum(a: uint256[2]) -> uint256:
    return a[0] + a[1]

@internal
def _driver(a: uint256[2]) -> uint256:
    tmp: uint256[2] = self._id(a)
    return self._sum(tmp)

@external
def foo() -> uint256:
    x: uint256[2] = [1, 2]
    return self._driver(x)
    """

    settings = Settings(experimental_codegen=True, optimize=OptimizationLevel.O3)
    settings.venom_flags = VenomOptimizationFlags(level=OptimizationLevel.O3, disable_inlining=True)

    ctx = compile_code(code, settings=settings, output_formats=["ir_runtime"])["ir_runtime"]

    driver_fn = next(fn for fn in ctx.functions.values() if "_driver" in str(fn.name))
    opcodes = [inst.opcode for bb in driver_fn.get_basic_blocks() for inst in bb.instructions]

    # Ensure the test is meaningful: _driver should still perform both calls.
    assert opcodes.count("invoke") >= 2
    # tmp should bind to _id's return buffer, not emit an intermediate copy.
    assert "mcopy" not in opcodes
