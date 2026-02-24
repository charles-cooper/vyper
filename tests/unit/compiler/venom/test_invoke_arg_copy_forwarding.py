from tests.venom_utils import parse_venom
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.basicblock import IRLabel, IRVariable
from vyper.venom.passes import InvokeArgCopyForwardingPass, ReadonlyMemoryArgsAnalysisPass


def _run_copy_forwarding(src: str):
    ctx = parse_venom(src)
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
