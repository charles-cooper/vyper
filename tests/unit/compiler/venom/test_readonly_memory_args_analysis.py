from tests.venom_utils import parse_venom
from vyper.venom.analysis import IRAnalysesCache
from vyper.venom.basicblock import IRLabel
from vyper.venom.passes import ReadonlyMemoryArgsAnalysisPass


def _run_readonly_analysis(src: str):
    ctx = parse_venom(src)
    analyses = {fn: IRAnalysesCache(fn) for fn in ctx.functions.values()}
    ReadonlyMemoryArgsAnalysisPass(analyses, ctx).run_pass()
    return ctx


def test_gep_write_marks_param_mutable():
    src = """
    function f {
    f:
        %arg = param
        %retpc = param
        %ptr = gep 32, %arg
        mstore %ptr, 1
        ret %retpc
    }
    """

    ctx = _run_readonly_analysis(src)
    fn = ctx.get_function(IRLabel("f"))
    assert fn._readonly_memory_invoke_arg_idxs == ()


def test_gep_read_keeps_param_readonly():
    src = """
    function f {
    f:
        %arg = param
        %retpc = param
        %ptr = gep 32, %arg
        %val = mload %ptr
        ret %retpc
    }
    """

    ctx = _run_readonly_analysis(src)
    fn = ctx.get_function(IRLabel("f"))
    assert fn._readonly_memory_invoke_arg_idxs == (0,)
