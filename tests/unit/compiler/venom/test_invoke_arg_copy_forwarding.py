from tests.venom_utils import parse_venom
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.basicblock import IRLabel, IRVariable
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


def test_readonly_forwarding_rejects_same_source_mutable_sibling_arg():
    src = """
    function caller {
    caller:
        %src = alloca 32
        %tmp = alloca 32
        mcopy %tmp, %src, 32
        invoke @callee, %tmp, %src
        stop
    }

    function callee {
    callee:
        %arg_ro = param
        %arg_rw = param
        %retpc = param
        mstore %arg_rw, 1
        %v = mload %arg_ro
        mstore 0, %v
        ret %retpc
    }
    """

    ctx = _run_copy_forwarding(src)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    # Forwarding arg_ro -> %src would alias it with mutable arg_rw.
    invoke = next(inst for inst in insts if inst.opcode == "invoke")
    invoke_args = list(invoke.operands[1:])
    assert invoke_args.count(IRVariable("%tmp")) == 1
    assert invoke_args.count(IRVariable("%src")) == 1
    assert any(inst.opcode == "mcopy" for inst in insts)


def test_readonly_forwarding_allows_same_source_readonly_sibling_arg():
    src = """
    function caller {
    caller:
        %src = alloca 32
        %tmp = alloca 32
        mcopy %tmp, %src, 32
        invoke @callee, %tmp, %src
        stop
    }

    function callee {
    callee:
        %arg0 = param
        %arg1 = param
        %retpc = param
        %v0 = mload %arg0
        %v1 = mload %arg1
        mstore 0, %v0
        mstore 32, %v1
        ret %retpc
    }
    """

    ctx = _run_copy_forwarding(src)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    # Both callee args are readonly, so aliasing the two args is safe.
    invoke = next(inst for inst in insts if inst.opcode == "invoke")
    invoke_args = list(invoke.operands[1:])
    assert invoke_args.count(IRVariable("%tmp")) == 0
    assert invoke_args.count(IRVariable("%src")) == 2
    assert all(inst.opcode != "mcopy" for inst in insts)


def test_internal_return_forwarding_still_applies_without_src_clobber():
    src = """
    function caller {
    caller:
        %src = alloca 32
        %dst = alloca 32
        invoke @callee, %src
        mcopy %dst, %src, 32
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
        callee._invoke_param_count = 1
        callee._has_memory_return_buffer_param = True

    ctx = _run_copy_forwarding(src, setup=_setup)
    caller = ctx.get_function(IRLabel("caller"))
    insts = [inst for bb in caller.get_basic_blocks() for inst in bb.instructions]

    # Forwarding should remove the copy and rewrite the load to source buffer.
    assert all(inst.opcode != "mcopy" for inst in insts)
    mload = next(inst for inst in insts if inst.opcode == "mload")
    assert mload.operands[0] == IRVariable("%src")
