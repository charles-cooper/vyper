
def safe_add(x: IRnode, y: IRnode):
    # precondition: x.typ.typ == t.typ.typ

    typinfo = x.typ._int_info if is_integer_type(x.typ) else x.typ._decimal_info

    res = IRnode.from_list(["add", x, y], typ=x.typ.typ)

    if typinfo.bits < 256:
        return clamp_basetype(res)

    # bits == 256
    with res.cache_when_complex("ans") as (b1, res):
        if typinfo.is_signed:
            # if r < 0:
            #   ans < l
            # else:
            #   ans >= l  # aka (iszero (ans < l))
            # aka: (r < 0) == (ans < l)
            clamp = ["eq", ["slt", y, 0], ["slt", res, x]]
        else:
            # note this is "equivalent" to the unsigned form
            # of the above (because y < 0 == False)
            #       ["eq", ["lt", y, 0], ["lt", res, x]]
            clamp = ["ge", res, x]

        return b1.resolve(["seq", clamp, res])

    raise CompilerPanic("unreachable")  # pragma: notest


def safe_sub(x: IRnode, y: IRnode):
    typinfo = x.typ._int_info if is_integer_type(x.typ) else x.typ._decimal_info

    res = IRnode.from_list(["sub", x, y], typ=x.typ.typ)

    if typinfo.bits < 256:
        return clamp_basetype(res)

    # bits == 256
    with res.cache_when_complex("ans") as (b1, res):
        if typinfo.is_signed:
            # if r < 0:
            #   ans > l
            # else:
            #   ans <= l  # aka (iszero (ans > l))
            # aka: (r < 0) == (ans > l)
            clamp = ["eq", ["slt", y, 0], ["sgt", res, x]]
        else:
            # note this is "equivalent" to the unsigned form
            # of the above (because y < 0 == False)
            #       ["eq", ["lt", y, 0], ["gt", res, x]]
            clamp = ["le", res, x]

        return b1.resolve(["seq", ["assert", clamp], res])

    raise CompilerPanic("unreachable")  # pragma: notest


def safe_mul(x: IRnode, y: IRnode):
    # precondition: x.typ.typ == y.typ.typ

    res = IRnode.from_list(["mul", x, y], typ=x.typ.typ)

    with res.cache_when_complex("ans") as (b1, res):

        ok = None

        if bits > 128: # check overflow mod 256
            # assert (res/l == r || l == 0)
            DIV = "sdiv" if signed else "div"
            ok = ["or", ["eq", [DIV, res, x], y], ["iszero", x]]

        if x.typ.typ == "int256":
            # special case:
            # in the sdiv check, if (l==-1 and r==-2**255),
            # -2**255<res> / -1<l> will return -2**255<r>.
            # need to check for this case.
            if version_check(begin="constantinople"):
                upper_bound = ["shl", 255, 1]
            else:
                upper_bound = -(2 ** 255)

            if not left.is_literal and not right.is_literal:
                # TODO can simplify this condition?
                bounds_check = ["or", ["ne", "l", ["not", 0]], ["ne", "r", upper_bound]]

            # TODO push some of this constant folding into optimizer
            elif left.is_literal and left.value == -1:
                bounds_check = ["ne", "r", upper_bound]
            elif right.is_literal and right.value == -(2 ** 255):
                bounds_check = ["ne", "l", ["not", 0]]
            else:
                bounds_check = "pass"

            ok = ["and", bounds_check, ok]

        # check overflow mod <bits>
        # NOTE: if 128 < bits < 256, `x * y` could be between
        # MAX_<bits> and 2**256 OR it could overflow past 2**256. so,
        # we check for overflow in mod 256 AND mod <bits>
        # (if bits == 256, clamp_basetype is a no-op)
        res = clamp_basetype(res)

        clamp = ["pass"] if ok is None else ["assert", ok]

        res = IRnode.from_list(["seq", clamp, res], typ=res.typ)

        return b1.resolve(res)


def safe_div(x: IRnode, y: IRnode):
    if y.value == 0:
        return  # raises

    clamp = ["pass"]

    # TODO: refactor this condition / push some things into the optimizer
    if ltyp == "int256":
        if version_check(begin="constantinople"):
            upper_bound = ["shl", 255, 1]
        else:
            upper_bound = -(2 ** 255)
        if not left.is_literal and not right.typ.is_literal:
            clamp = ["assert", ["or", ["ne", x, ["not", 0]], ["ne", y, upper_bound]]]
        # TODO push this constant folding into the optimizer
        elif left.is_literal and left.value == -(2 ** 255):
            clamp = ["assert", ["ne", x, ["not", 0]]]
        elif right.is_literal and right.value == -1:
            clamp = ["assert", ["ne", y, upper_bound]]

    DIV = "sdiv" if signed else "div"
    return ["seq", clamp, [DIV, x, ["clamp_nonzero", y]]]


def safe_mod(x: IRnode, y: IRnode):
    if y.value == 0:
        return  # raises
    MOD = "smod" if signed else "mod"
    return ["seq", [MOD, x, ["clamp_nonzero", y]]]


def safe_pow(x: IRnode, y: IRnode):
    pass
