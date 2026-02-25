from tests.venom_utils import parse_venom
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.basicblock import IRLabel, IRLiteral, IRVariable
from vyper.venom.passes import InvokeArgCopyForwardingPass, ReadonlyMemoryArgsAnalysisPass


def _run_copy_forwarding(src: str, setup=None):
    ctx = parse_venom(src)
    if setup is not None:
        setup(ctx)
    analyses = {fn: IRAnalysesCache(fn) for fn in ctx.functions.values()}
    ReadonlyMemoryArgsAnalysisPass(analyses, ctx).run_pass()
    for fn in ctx.functions.values():
        InvokeArgCopyForwardingPass(analyses[fn], fn).run_pass()
    return ctx


def test_readonly_forwarding_rejects_src_clobber_before_invoke():
    src = """
    function caller {
    caller:
        %src = alloca 64
        %tmp = alloca 64
        mcopy %tmp, %src, 64
        mstore %src, 1
        invoke @callee, %tmp
        stop
    }

    function callee {
    callee:
        %arg = param
        %retpc = param
        mload %arg
        ret %retpc
    }
    """

    ctx = _run_copy_forwarding(src)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    mcopy = next(inst for inst in insts if inst.opcode == "mcopy")
    invoke = next(inst for inst in insts if inst.opcode == "invoke")

    # Copy must remain; invoke should still use the staged destination.
    assert invoke.operands[1] == mcopy.operands[2]


def test_readonly_forwarding_still_applies_without_src_clobber():
    src = """
    function caller {
    caller:
        %src = alloca 64
        %tmp = alloca 64
        mcopy %tmp, %src, 64
        invoke @callee, %tmp
        stop
    }

    function callee {
    callee:
        %arg = param
        %retpc = param
        mload %arg
        ret %retpc
    }
    """

    ctx = _run_copy_forwarding(src)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    invoke = next(inst for inst in insts if inst.opcode == "invoke")
    assert invoke.operands[1] == IRVariable("%src")
    assert all(inst.opcode != "mcopy" for inst in insts)


def test_internal_return_forwarding_rejects_clobber_between_copy_and_use():
    src = """
    function caller {
    caller:
        %src = alloca 32
        %dst = alloca 32
        invoke @callee, %src
        mcopy %dst, %src, 32
        mstore 64, 1
        %v = mload %dst
        sink %v
    }

    function callee {
    callee:
        %retbuf = param
        %retpc = param
        mstore %retbuf, 7
        ret %retpc
    }
    """

    def _setup(ctx):
        callee = ctx.get_function(IRLabel("callee"))
        # Memory-return internal function: first invoke arg is return buffer.
        callee._invoke_param_count = 1
        callee._has_memory_return_buffer_param = True
        # Force current pass lookup path (`ctx.functions.get(target.value)`) to
        # resolve the callee so this test exercises internal-return forwarding.
        ctx.functions["callee"] = callee

    ctx = _run_copy_forwarding(src, setup=_setup)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    # Ensure the test shape still has the potential clobbering write.
    mstore = next(inst for inst in insts if inst.opcode == "mstore")
    assert mstore.operands[1] == IRLiteral(64)

    # This copy must remain: %dst is a snapshot of %src before the clobber.
    mcopy = next(inst for inst in insts if inst.opcode == "mcopy")
    mload = next(inst for inst in insts if inst.opcode == "mload")
    assert mload.operands[0] == mcopy.operands[2]
