import operator
from typing import Any, List, Optional

from vyper.parser.parser_utils import LLLnode
from vyper.utils import LOADED_LIMITS
import math

import copy


def get_int_at(args: List[LLLnode], pos: int, signed: bool = False) -> Optional[int]:
    value = args[pos].value

    if isinstance(value, int):
        o = value
    elif (
        value == "mload"
        and args[pos].args[0].value in LOADED_LIMITS.keys()
        and isinstance(args[pos].args[0].value, int)
    ):
        idx = int(args[pos].args[0].value)  # isinstance in if confirms type is int.
        o = LOADED_LIMITS[idx]
    else:
        return None

    if signed or o < 0:
        return ((o + 2 ** 255) % 2 ** 256) - 2 ** 255
    else:
        return o % 2 ** 256


def int_at(args: List[LLLnode], pos: int, signed: bool = False) -> Optional[int]:
    return get_int_at(args, pos, signed) is not None


arith = {
    "add": (operator.add, "+"),
    "sub": (operator.sub, "-"),
    "mul": (operator.mul, "*"),
    "div": (operator.floordiv, "/"),
    "mod": (operator.mod, "%"),
}


def _is_constant_add(node: LLLnode, args: List[LLLnode]) -> bool:
    return bool(
        (isinstance(node.value, str))
        and (node.value == "add" and int_at(args, 0))
        and (args[1].value == "add" and int_at(args[1].args, 0))
    )


def optimize(lll_node: LLLnode) -> LLLnode:
    lll_node = apply_general_optimizations(lll_node)
    lll_node = filter_unused_sizelimits(lll_node)
    lll_node = merge_push_constants(lll_node)

    return lll_node


def apply_general_optimizations(node: LLLnode) -> LLLnode:
    # TODO refactor this into several functions
    argz = [apply_general_optimizations(arg) for arg in node.args]

    if node.value == "seq":
        _merge_memzero(argz)
        _merge_calldataload(argz)

    if node.value in arith and int_at(argz, 0) and int_at(argz, 1):
        left, right = get_int_at(argz, 0), get_int_at(argz, 1)
        # `node.value in arith` implies that `node.value` is a `str`
        calcer, symb = arith[str(node.value)]
        new_value = calcer(left, right)
        if argz[0].annotation and argz[1].annotation:
            annotation = argz[0].annotation + symb + argz[1].annotation
        elif argz[0].annotation or argz[1].annotation:
            annotation = (
                (argz[0].annotation or str(left)) + symb + (argz[1].annotation or str(right))
            )
        else:
            annotation = ""
        return LLLnode(
            new_value,
            [],
            node.typ,
            None,
            node.pos,
            annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    elif _is_constant_add(node, argz):
        # `node.value in arith` implies that `node.value` is a `str`
        calcer, symb = arith[str(node.value)]
        if argz[0].annotation and argz[1].args[0].annotation:
            annotation = argz[0].annotation + symb + argz[1].args[0].annotation
        elif argz[0].annotation or argz[1].args[0].annotation:
            annotation = (
                (argz[0].annotation or str(argz[0].value))
                + symb
                + (argz[1].args[0].annotation or str(argz[1].args[0].value))
            )
        else:
            annotation = ""
        return LLLnode(
            "add",
            [
                LLLnode(int(argz[0].value) + int(argz[1].args[0].value), annotation=annotation),
                argz[1].args[1],
            ],
            node.typ,
            None,
            annotation=node.annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    elif node.value == "add" and get_int_at(argz, 0) == 0:
        return LLLnode(
            argz[1].value,
            argz[1].args,
            node.typ,
            node.location,
            node.pos,
            annotation=argz[1].annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    elif node.value == "add" and get_int_at(argz, 1) == 0:
        return LLLnode(
            argz[0].value,
            argz[0].args,
            node.typ,
            node.location,
            node.pos,
            argz[0].annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    elif node.value == "clamp" and int_at(argz, 0) and int_at(argz, 1) and int_at(argz, 2):
        if get_int_at(argz, 0, True) > get_int_at(argz, 1, True):  # type: ignore
            raise Exception("Clamp always fails")
        elif get_int_at(argz, 1, True) > get_int_at(argz, 2, True):  # type: ignore
            raise Exception("Clamp always fails")
        else:
            return argz[1]
    elif node.value == "clamp" and int_at(argz, 0) and int_at(argz, 1):
        if get_int_at(argz, 0, True) > get_int_at(argz, 1, True):  # type: ignore
            raise Exception("Clamp always fails")
        else:
            return LLLnode(
                "clample",
                [argz[1], argz[2]],
                node.typ,
                node.location,
                node.pos,
                node.annotation,
                add_gas_estimate=node.add_gas_estimate,
                valency=node.valency,
            )
    elif node.value == "clamp_nonzero" and int_at(argz, 0):
        if get_int_at(argz, 0) != 0:
            return LLLnode(
                argz[0].value,
                [],
                node.typ,
                node.location,
                node.pos,
                node.annotation,
                add_gas_estimate=node.add_gas_estimate,
                valency=node.valency,
            )
        else:
            raise Exception("Clamp always fails")
    # [eq, x, 0] is the same as [iszero, x].
    elif node.value == "eq" and int_at(argz, 1) and argz[1].value == 0:
        return LLLnode(
            "iszero",
            [argz[0]],
            node.typ,
            node.location,
            node.pos,
            node.annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    # [ne, x, y] has the same truthyness as [xor, x, y]
    # rewrite 'ne' as 'xor' in places where truthy is accepted.
    elif node.value in ("if", "if_unchecked", "assert") and argz[0].value == "ne":
        argz[0] = LLLnode.from_list(["xor"] + argz[0].args)  # type: ignore
        return LLLnode.from_list(
            [node.value] + argz,  # type: ignore
            typ=node.typ,
            location=node.location,
            pos=node.pos,
            annotation=node.annotation,
            # let from_list handle valency and gas_estimate
        )
    elif node.value == "seq":
        xs: List[Any] = []
        for arg in argz:
            if arg.value == "seq":
                xs.extend(arg.args)
            else:
                xs.append(arg)
        return LLLnode(
            node.value,
            xs,
            node.typ,
            node.location,
            node.pos,
            node.annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
    elif node.total_gas is not None:
        o = LLLnode(
            node.value,
            argz,
            node.typ,
            node.location,
            node.pos,
            node.annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )
        o.total_gas = node.total_gas - node.gas + o.gas
        o.func_name = node.func_name
        return o
    else:
        return LLLnode(
            node.value,
            argz,
            node.typ,
            node.location,
            node.pos,
            node.annotation,
            add_gas_estimate=node.add_gas_estimate,
            valency=node.valency,
        )


def _merge_memzero(argz):
    # look for sequential mzero / calldatacopy operations that are zero'ing memory
    # and merge them into a single calldatacopy
    mstore_nodes: List = []
    initial_offset = 0
    total_length = 0
    for lll_node in [i for i in argz if i.value != "pass"]:
        if (
            lll_node.value == "mstore"
            and isinstance(lll_node.args[0].value, int)
            and lll_node.args[1].value == 0
        ):
            # mstore of a zero value
            offset = lll_node.args[0].value
            if not mstore_nodes:
                initial_offset = offset
            if initial_offset + total_length == offset:
                mstore_nodes.append(lll_node)
                total_length += 32
                continue

        if (
            lll_node.value == "calldatacopy"
            and isinstance(lll_node.args[0].value, int)
            and lll_node.args[1].value == "calldatasize"
            and isinstance(lll_node.args[2].value, int)
        ):
            # calldatacopy from the end of calldata - efficient zero'ing via `empty()`
            offset, length = lll_node.args[0].value, lll_node.args[2].value
            if not mstore_nodes:
                initial_offset = offset
            if initial_offset + total_length == offset:
                mstore_nodes.append(lll_node)
                total_length += length
                continue

        # if we get this far, the current node is not a zero'ing operation
        # it's time to apply the optimization if possible
        if len(mstore_nodes) > 1:
            new_lll = LLLnode.from_list(
                ["calldatacopy", initial_offset, "calldatasize", total_length],
                pos=mstore_nodes[0].pos,
            )
            # replace first zero'ing operation with optimized node and remove the rest
            idx = argz.index(mstore_nodes[0])
            argz[idx] = new_lll
            for i in mstore_nodes[1:]:
                argz.remove(i)

        initial_offset = 0
        total_length = 0
        mstore_nodes.clear()


def _merge_calldataload(argz):
    # look for sequential operations copying from calldata to memory
    # and merge them into a single calldatacopy operation
    mstore_nodes: List = []
    initial_mem_offset = 0
    initial_calldata_offset = 0
    total_length = 0
    for lll_node in [i for i in argz if i.value != "pass"]:
        if (
            lll_node.value == "mstore"
            and isinstance(lll_node.args[0].value, int)
            and lll_node.args[1].value == "calldataload"
            and isinstance(lll_node.args[1].args[0].value, int)
        ):
            # mstore of a zero value
            mem_offset = lll_node.args[0].value
            calldata_offset = lll_node.args[1].args[0].value
            if not mstore_nodes:
                initial_mem_offset = mem_offset
                initial_calldata_offset = calldata_offset
            if (
                initial_mem_offset + total_length == mem_offset
                and initial_calldata_offset + total_length == calldata_offset
            ):
                mstore_nodes.append(lll_node)
                total_length += 32
                continue

        # if we get this far, the current node is a different operation
        # it's time to apply the optimization if possible
        if len(mstore_nodes) > 1:
            new_lll = LLLnode.from_list(
                ["calldatacopy", initial_mem_offset, initial_calldata_offset, total_length],
                pos=mstore_nodes[0].pos,
            )
            # replace first copy operation with optimized node and remove the rest
            idx = argz.index(mstore_nodes[0])
            argz[idx] = new_lll
            for i in mstore_nodes[1:]:
                argz.remove(i)

        initial_mem_offset = 0
        initial_calldata_offset = 0
        total_length = 0
        mstore_nodes.clear()


def filter_unused_sizelimits(lll_node: LLLnode) -> LLLnode:
    # recursively search the LLL for mloads of the size limits, and then remove
    # the initial mstore operations for size limits that are never referenced
    expected_offsets = set(LOADED_LIMITS)
    seen_offsets = _find_mload_offsets(lll_node, expected_offsets, set())
    if expected_offsets == seen_offsets:
        return lll_node

    unseen_offsets = expected_offsets.difference(seen_offsets)
    _remove_mstore(lll_node, unseen_offsets)

    return lll_node


def _find_mload_offsets(lll_node: LLLnode, expected_offsets: set, seen_offsets: set) -> set:
    for node in lll_node.args:
        if node.value == "mload" and node.args[0].value in expected_offsets:
            location = next(i for i in expected_offsets if i == node.args[0].value)
            seen_offsets.add(location)
        else:
            seen_offsets.update(_find_mload_offsets(node, expected_offsets, seen_offsets))

    return seen_offsets

# search for repeated constants PUSHed onto the stack and
# turn them into `with` statements (so future references can just use dup)
def merge_push_constants(lll_node: LLLnode, max_slots: int = 16) -> LLLnode:
    lll_node = copy.deepcopy(lll_node)

    # search for the max 'with' depth so that we don't use more than 16
    # stack slots
    def _maxheight(x: LLLnode):
        # "lll" is an optimization barrier.
        xs = list(_maxheight(a) for a in x.args if a.value != "lll")
        mh = max(xs) if xs else 0
        if x.value == 'with':
            return mh + 1
        else:
            return mh

    available_slots = max(0, max_slots - _maxheight(lll_node))

    # now search for the most bang for our buck.
    def _constants(x: LLLnode, consts_list: dict=None) -> dict:
        if consts_list is None:
            consts_list = {}

        # find constants.
        # nice-to-have: add jumpdest locations into the set
        if isinstance(x.value, int):
            consts_list.setdefault(x.value, 1)
            consts_list[x.value] += 1
        else:
            for a in x.args:
                _constants(a, consts_list)

        return consts_list

    consts = _constants(lll_node)
    # translate constant into bytes used in the bytecode
    # PUSH1 x   -> 1
    # PUSH2 xx  -> 2
    # PUSH3 xxx -> 3
    # and so on
    bytecode_overhead = lambda x: int(math.log(x, 256)) + 1
    offenders = [(k, bytecode_overhead(v)) for k, v in consts.items()]

    # get the top K offenders by bytecode_overhead
    offenders.sort(key=lambda x: x[1])[:available_slots]
    offenders = dict(offenders)

    def _const_to_varname(x: int):
        return f"CONST_{x}"

    ret = lll_node
    # replace constants in place
    def _replace_constants(x: LLLnode):
        if isinstance(x.value, int) and x.value in to_use:
            x.value = _const_to_varname(x.value)
        # "lll" is an optimization barrier.
        if x.value != "lll":
            for x in x.args:
                _replace_constants(x)
        else:
            x.args[0] = merge_push_constants(x.args[0])

    _replace_constants(ret)

    for x in offenders:
        varname = _const_to_varname(x)
        ret = LLLnode.from_list(["with", varname, x, ret])

    return ret


def _remove_mstore(lll_node: LLLnode, offsets: set) -> None:
    for node in lll_node.args.copy():
        if node.value == "mstore" and node.args[0].value in offsets:
            lll_node.args.remove(node)
        else:
            _remove_mstore(node, offsets)
