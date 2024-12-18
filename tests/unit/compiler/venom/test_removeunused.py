from tests.venom_utils import assert_ctx_eq, parse_from_basic_block, parse_venom
from vyper.venom.analysis.analysis import IRAnalysesCache
from vyper.venom.passes import RemoveUnusedVariablesPass


def _check_pre_post(pre, post, scope="basicblock"):
    if scope == "basicblock":
        parse = parse_from_basic_block
    else:
        parse = parse_venom

    ctx = parse(pre)
    for fn in ctx.functions.values():
        ac = IRAnalysesCache(fn)
        RemoveUnusedVariablesPass(ac, fn).run_pass()

    assert_ctx_eq(ctx, parse(post))


def _check_no_change(pre, scope="basicblock"):
    _check_pre_post(pre, pre, scope=scope)


def test_removeunused_basic():
    """
    Check basic unused variable removal
    """
    pre = """
    main:
        %1 = add 10, 20
        %2_unused = add 10, %1
        mstore 20, %1
        stop
    """
    post = """
    main:
        %1 = add 10, 20
        mstore 20, %1
        stop
    """
    _check_pre_post(pre, post)


def test_removeunused_msize_basic():
    pre = """
    main:
        %a = mload 32
        %b = msize
        %c_unused = mload 64  # safe to remove
        return %b, %b
    """
    post = """
    main:
        %a = mload 32
        %b = msize
        return %b, %b
    """
    _check_pre_post(pre, post)


def test_removeunused_msize_two_msizes():
    pre = """
    main:
        %a = mload 32
        %b = msize
        %c = mload 64  # not safe to remove - has MSIZE effect
        %d = msize
        return %b, %d
    """
    _check_no_change(pre)


def test_removeunused_msize_loop():
    pre = """
    main:
        %msize = msize

        # not safe to remove because the previous instruction is
        # still reachable
        %1 = mload %msize
        jmp @main
    """
    _check_no_change(pre)


def test_removeunused_msize_branches():
    pre = """
    function global {
        main:
            %1 = param
            %2 = mload 10  ; looks unused, but has MSIZE effect
            jnz %1, @branch1, @branch2
        branch1:
            %3 = msize  ; blocks removal of `%2 = mload 10`
            mstore 10, %3
            jmp @end
        branch2:
            jmp @end
        end:
            stop
    }
    """
    _check_no_change(pre, scope="function")


def test_removeunused_unused_msize_loop():
    pre = """
    main:
        %1_unused = msize
        %2_unused = mload 10
        jmp @main
    """
    post = """
    main:
        jmp @main
    """
    _check_pre_post(pre, post)


def test_removeunused_unused_msize():
    pre = """
    main:
        %1_unused = mload 10
        %2_unused = msize
        stop
    """
    post = """
    main:
        stop
    """
    _check_pre_post(pre, post)


def test_removeunused_loop():
    pre = """
    main:
        %1 = 10
        jmp @after
    after:
        %p = phi @main, %1, @after, %2
        %2 = add %p, 1
        %3_unused = add %2, %p
        mstore 10, %2
        jmp @after
    """
    post = """
    main:
        %1 = 10
        jmp @after
    after:
        %p = phi @main, %1, @after, %2
        %2 = add %p, 1
        mstore 10, %2
        jmp @after
    """
    _check_pre_post(pre, post)