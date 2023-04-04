from typing import Any, List

import vyper.utils as util
from vyper.address_space import CALLDATA, DATA, MEMORY
from vyper.ast.signatures.function_signature import FunctionSignature
from vyper.codegen.abi_encoder import abi_encoding_matches_vyper
from vyper.codegen.context import Context, VariableRecord
from vyper.codegen.core import get_element_ptr, getpos, make_setter, needs_clamp
from vyper.codegen.expr import Expr
from vyper.codegen.function_definitions.utils import get_nonreentrant_lock
from vyper.codegen.ir_node import Encoding, IRnode
from vyper.codegen.stmt import parse_body
from vyper.semantics.types import TupleT


# register function args with the local calling context.
# also allocate the ones that live in memory (i.e. kwargs)
def _register_function_args(context: Context) -> List[IRnode]:
    ret = []

    # the type of the calldata
    base_args_t = TupleT(tuple(arg.typ for arg in sig.base_args))

    # tuple with the abi_encoded args
    if context.func_t.is_init_func:
        base_args_ofst = IRnode(0, location=DATA, typ=base_args_t, encoding=Encoding.ABI)
    else:
        base_args_ofst = IRnode(4, location=CALLDATA, typ=base_args_t, encoding=Encoding.ABI)

    for i, arg in enumerate(sig.base_args):
        arg_ir = get_element_ptr(base_args_ofst, i)

        if needs_clamp(arg.typ, Encoding.ABI):
            # allocate a memory slot for it and copy
            p = context.new_variable(arg.name, arg.typ, is_mutable=False)
            dst = IRnode(p, typ=arg.typ, location=MEMORY)

            copy_arg = make_setter(dst, arg_ir)
            copy_arg.source_pos = getpos(arg.ast_source)
            ret.append(copy_arg)
        else:
            assert abi_encoding_matches_vyper(arg.typ)
            # leave it in place
            context.vars[arg.name] = VariableRecord(
                name=arg.name,
                pos=arg_ir,
                typ=arg.typ,
                mutable=False,
                location=arg_ir.location,
                encoding=Encoding.ABI,
            )

    return ret


def _annotated_method_id(abi_sig):
    method_id = util.method_id_int(abi_sig)
    annotation = f"{hex(method_id)}: {abi_sig}"
    return IRnode(method_id, annotation=annotation)


def _generate_kwarg_handlers(context: Context) -> List[Any]:
    # generate kwarg handlers.
    # since they might come in thru calldata or be default,
    # allocate them in memory and then fill it in based on calldata or default,
    # depending on the signature
    # a kwarg handler looks like
    # (if (eq _method_id <method_id>)
    #    copy calldata args to memory
    #    write default args to memory
    #    goto external_function_common_ir

    def handler_for(calldata_kwargs, default_kwargs):
        calldata_args = func_t.base_args + calldata_kwargs
        # create a fake type so that get_element_ptr works
        calldata_args_t = TupleT(list(arg.typ for arg in calldata_args))

        abi_sig = func_t.abi_signature_for_kwargs(calldata_kwargs)
        method_id = _annotated_method_id(abi_sig)

        calldata_kwargs_ofst = IRnode(
            4, location=CALLDATA, typ=calldata_args_t, encoding=Encoding.ABI
        )

        # a sequence of statements to strictify kwargs into memory
        ret = ["seq"]

        # ensure calldata is at least of minimum length
        args_abi_t = calldata_args_t.abi_type
        calldata_min_size = args_abi_t.min_size() + 4
        ret.append(["assert", ["ge", "calldatasize", calldata_min_size]])

        # TODO optimize make_setter by using
        # TupleT(list(arg.typ for arg in calldata_kwargs + default_kwargs))
        # (must ensure memory area is contiguous)

        n_base_args = len(func_t.base_args)

        for i, arg_meta in enumerate(calldata_kwargs):
            k = n_base_args + i

            dst = context.lookup_var(arg_meta.name).pos

            lhs = IRnode(dst, location=MEMORY, typ=arg_meta.typ)

            rhs = get_element_ptr(calldata_kwargs_ofst, k, array_bounds_check=False)

            copy_arg = make_setter(lhs, rhs)
            copy_arg.source_pos = getpos(arg_meta.ast_source)
            ret.append(copy_arg)

        for x in default_kwargs:
            dst = context.lookup_var(x.name).pos
            lhs = IRnode(dst, location=MEMORY, typ=x.typ)
            lhs.source_pos = getpos(x.ast_source)
            kw_ast_val = sig.default_values[x.name]  # e.g. `3` in x: int = 3
            rhs = Expr(kw_ast_val, context).ir_node

            copy_arg = make_setter(lhs, rhs)
            copy_arg.source_pos = getpos(x.ast_source)
            ret.append(copy_arg)

        ret.append(["goto", sig.external_function_base_entry_label])

        method_id_check = ["eq", "_calldata_method_id", method_id]

        # if there is a function whose selector is 0, it won't be distinguished
        # from the case where nil calldata is supplied, b/c calldataload loads
        # 0s past the end of physical calldata (cf. yellow paper).
        # since supplying 0 calldata is expected to trigger the fallback fn,
        # we check that calldatasize > 0, which distinguishes the 0 selector
        # from the fallback function "selector"
        # (equiv. to "all selectors not in the selector table").

        # note: cases where not enough calldata is supplied (besides
        # calldatasize==0) are not addressed here b/c a calldatasize
        # well-formedness check is already present in the function body
        # as part of abi validation
        if method_id.value == 0:
            method_id_check = ["and", ["gt", "calldatasize", 0], method_id_check]

        ret = ["if", method_id_check, ret]
        return ret

    ret = ["seq"]

    keyword_args = sig.default_args

    # allocate variable slots in memory
    for arg in keyword_args:
        context.new_variable(arg.name, arg.typ, is_mutable=False)

    for i, _ in enumerate(keyword_args):
        calldata_kwargs = keyword_args[:i]
        default_kwargs = keyword_args[i:]

        ret.append(handler_for(calldata_kwargs, default_kwargs))

    ret.append(handler_for(keyword_args, []))

    return ret


# TODO it would be nice if this returned a data structure which were
# amenable to generating a jump table instead of the linear search for
# method_id we have now.
def generate_ir_for_external_function(code, context, skip_nonpayable_check):
    # TODO type hints:
    # def generate_ir_for_external_function(
    #    code: vy_ast.FunctionDef, sig: FunctionSignature, context: Context, check_nonpayable: bool,
    # ) -> IRnode:
    """Return the IR for an external function. Includes code to inspect the method_id,
    enter the function (nonpayable and reentrancy checks), handle kwargs and exit
    the function (clean up reentrancy storage variables)
    """
    func_type = code._metadata["type"]

    nonreentrant_pre, nonreentrant_post = get_nonreentrant_lock(func_type)

    # generate handlers for base args and register the variable records
    handle_base_args = _register_function_args(context, sig)

    # generate handlers for kwargs and register the variable records
    kwarg_handlers = _generate_kwarg_handlers(context, sig)

    body = ["seq"]
    # once optional args have been handled,
    # generate the main body of the function
    body += handle_base_args

    if sig.mutability != "payable" and not skip_nonpayable_check:
        # if the contract contains payable functions, but this is not one of them
        # add an assertion that the value of the call is zero
        body += [["assert", ["iszero", "callvalue"]]]

    body += nonreentrant_pre

    body += [parse_body(code.body, context, ensure_terminated=True)]

    # wrap the body in labeled block
    body = ["label", sig.external_function_base_entry_label, ["var_list"], body]

    exit_sequence = ["seq"] + nonreentrant_post
    if sig.is_init_func:
        pass  # init func has special exit sequence generated by module.py
    elif context.return_type is None:
        exit_sequence += [["stop"]]
    else:
        exit_sequence += [["return", "ret_ofst", "ret_len"]]

    exit_sequence_args = ["var_list"]
    if context.return_type is not None:
        exit_sequence_args += ["ret_ofst", "ret_len"]
    # wrap the exit in a labeled block
    exit = ["label", sig.exit_sequence_label, exit_sequence_args, exit_sequence]

    # the ir which comprises the main body of the function,
    # besides any kwarg handling
    func_common_ir = ["seq", body, exit]

    if sig.is_default_func or sig.is_init_func:
        ret = ["seq"]
        # add a goto to make the function entry look like other functions
        # (for zksync interpreter)
        ret.append(["goto", sig.external_function_base_entry_label])
        ret.append(func_common_ir)
    else:
        ret = kwarg_handlers
        # sneak the base code into the kwarg handler
        # TODO rethink this / make it clearer
        ret[-1][-1].append(func_common_ir)

    return IRnode.from_list(ret, source_pos=getpos(sig.func_ast_code))
