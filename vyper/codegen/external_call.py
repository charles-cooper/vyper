from dataclasses import dataclass

import vyper.utils as util
from vyper.address_space import MEMORY
from vyper.codegen.abi_encoder import abi_encode
from vyper.codegen.core import (
    calculate_type_for_external_return,
    check_assign,
    check_external_call,
    dummy_node_for_type,
    make_setter,
    needs_clamp,
)
from vyper.codegen.ir_node import Encoding, IRnode
from vyper.codegen.types import InterfaceType, TupleType, get_type_for_exact_size
from vyper.codegen.types.convert import new_type_to_old_type
from vyper.exceptions import StateAccessViolation, TypeCheckFailure


@dataclass
class _SpecialKwargs:
    value: IRnode
    gas: IRnode
    skip_contract_check: bool
    if_empty_return_override: IRnode


def _pack_arguments(contract_sig, args, context):
    # abi encoding just treats all args as a big tuple
    args_tuple_t = TupleType([x.typ for x in args])
    args_as_tuple = IRnode.from_list(["multi"] + [x for x in args], typ=args_tuple_t)
    args_abi_t = args_tuple_t.abi_type

    # sanity typecheck - make sure the arguments can be assigned
    dst_tuple_t = TupleType([arg.typ for arg in contract_sig.args][: len(args)])
    check_assign(dummy_node_for_type(dst_tuple_t), args_as_tuple)

    if contract_sig.return_type is not None:
        return_abi_t = calculate_type_for_external_return(contract_sig.return_type).abi_type

        # we use the same buffer for args and returndata,
        # so allocate enough space here for the returndata too.
        buflen = max(args_abi_t.size_bound(), return_abi_t.size_bound())
    else:
        buflen = args_abi_t.size_bound()

    buflen += 32  # padding for the method id

    buf_t = get_type_for_exact_size(buflen)
    buf = context.new_internal_variable(buf_t)

    args_ofst = buf + 28
    args_len = args_abi_t.size_bound() + 4

    abi_signature = contract_sig.name + dst_tuple_t.abi_type.selector_name()

    # layout:
    # 32 bytes                 | args
    # 0x..00<method_id_4bytes> | args
    # the reason for the left padding is just so the alignment is easier.
    # if we were only targeting constantinople, we could align
    # to buf (and also keep code size small) by using
    # (mstore buf (shl signature.method_id 224))
    mstore_method_id = [["mstore", buf, util.abi_method_id(abi_signature)]]

    if len(args) == 0:
        encode_args = ["pass"]
    else:
        encode_args = abi_encode(buf + 32, args_as_tuple, context, bufsz=buflen)

    return buf, mstore_method_id + [encode_args], args_ofst, args_len


def _unpack_returndata(buf, contract_sig, skip_contract_check, return_override, context, expr):
    # expr.func._metadata["type"].return_type is more accurate
    # than contract_sig.return_type in the case of JSON interfaces.
    ast_return_t = expr.func._metadata["type"].return_type

    if ast_return_t is None:
        return ["pass"], 0, 0

    # sanity check
    return_t = new_type_to_old_type(ast_return_t)
    check_assign(dummy_node_for_type(return_t), dummy_node_for_type(contract_sig.return_type))

    return_t = calculate_type_for_external_return(return_t)

    abi_return_t = return_t.abi_type

    min_return_size = abi_return_t.min_size()
    max_return_size = abi_return_t.size_bound()
    assert 0 < min_return_size <= max_return_size

    ret_ofst = buf
    ret_len = max_return_size

    # revert when returndatasize is not in bounds
    ret = []
    # except when return_override is provided.
    # runtime: min_return_size <= returndatasize
    if not skip_contract_check:
        check = ["assert", ["ge", "returndatasize", min_return_size]]
    else:
        check = ["seq"]

    if return_override is not None:
        # if returndatasize == 0:
        #    copy return override to buf
        # else:
        #    assert returndatasize >= min_return_size
        ret.append(
            [
                "if",
                ["eq", "returndatasize", 0],
                abi_encode(buf, return_override, context, ret_len),
                check,
            ]
        )
    else:
        ret.append(check)

    encoding = Encoding.ABI

    buf = IRnode.from_list(
        buf,
        typ=return_t,
        location=MEMORY,
        encoding=encoding,
        annotation=f"{expr.node_source_code} returndata buffer",
    )

    assert isinstance(return_t, TupleType)
    # unpack strictly
    if needs_clamp(return_t, encoding):
        buf2 = IRnode.from_list(
            context.new_internal_variable(return_t), typ=return_t, location=MEMORY
        )

        ret.append(make_setter(buf2, buf))
        ret.append(buf2)
    else:
        ret.append(buf)

    return ret, ret_ofst, ret_len


def _external_call_helper(
    contract_address,
    contract_sig,
    args_ir,
    context,
    special_kwargs,
    expr=None,
):

    # sanity check
    assert len(contract_sig.base_args) <= len(args_ir) <= len(contract_sig.args)

    if context.is_constant() and contract_sig.mutability not in ("view", "pure"):
        # TODO is this already done in type checker?
        raise StateAccessViolation(
            f"May not call state modifying function '{contract_sig.name}' "
            f"within {context.pp_constancy()}.",
            expr,
        )

    sub = ["seq"]

    buf, arg_packer, args_ofst, args_len = _pack_arguments(contract_sig, args_ir, context)

    ret_unpacker, ret_ofst, ret_len = _unpack_returndata(
        buf,
        contract_sig,
        special_kwargs.skip_contract_check,
        special_kwargs.if_empty_return_override,
        context,
        expr,
    )

    sub += arg_packer

    if contract_sig.return_type is None and not special_kwargs.skip_contract_check:
        # if we do not expect return data, check that a contract exists at the
        # target address. we must perform this check BEFORE the call because
        # the contract might selfdestruct. on the other hand we can omit this
        # when we _do_ expect return data because we later check
        # `returndatasize` (that check works even if the contract
        # selfdestructs).
        sub.append(["assert", ["extcodesize", contract_address]])

    if context.is_constant() or contract_sig.mutability in ("view", "pure"):
        call_op = [
            "staticcall",
            special_kwargs.gas,
            contract_address,
            args_ofst,
            args_len,
            ret_ofst,
            ret_len,
        ]
    else:
        call_op = [
            "call",
            special_kwargs.gas,
            contract_address,
            special_kwargs.value,
            args_ofst,
            args_len,
            ret_ofst,
            ret_len,
        ]

    sub.append(check_external_call(call_op))

    if contract_sig.return_type is not None:
        sub += ret_unpacker

    return IRnode.from_list(sub, typ=contract_sig.return_type, location=MEMORY)


def _get_special_kwargs(call_expr, context):
    from vyper.codegen.expr import Expr  # TODO rethink this circular import

    def _bool(x):
        assert x in (0, 1), "type checker missed this"
        return bool(x)

    # turn kwargs into dict
    call_kwargs = {kw.arg: Expr.parse_value_expr(kw.value, context) for kw in call_expr.keywords}

    ret = _SpecialKwargs(
        value=IRnode.from_list(call_kwargs.pop("value", 0)),
        gas=IRnode.from_list(call_kwargs.pop("gas", "gas")),
        skip_contract_check=_bool(call_kwargs.pop("skip_contract_check", 0)),
        if_empty_return_override=call_kwargs.pop("if_empty_return_override"),
    )

    if call_kwargs != {}:
        raise TypeCheckFailure(f"Unexpected keyword arguments: {call_kwargs}")

    return ret


def ir_for_external_call(call_expr, context):
    from vyper.codegen.expr import Expr  # TODO rethink this circular import

    contract_address = Expr.parse_value_expr(call_expr.func.value, context)
    special_kwargs = _get_special_kwargs(call_expr, context)
    args_ir = [Expr(x, context).ir_node for x in call_expr.args]

    assert isinstance(contract_address.typ, InterfaceType)
    contract_name = contract_address.typ.name
    method_name = call_expr.func.attr
    contract_sig = context.sigs[contract_name][method_name]

    ret = _external_call_helper(
        contract_address,
        contract_sig,
        args_ir,
        context,
        special_kwargs=special_kwargs,
        expr=call_expr,
    )

    return ret
