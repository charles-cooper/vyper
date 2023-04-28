from math import ceil

from vyper.codegen.core import bytes_data_ptr, ensure_in_memory, get_bytearray_length
from vyper.codegen.ir_node import IRnode
from vyper.exceptions import CompilerPanic
from vyper.semantics.types.bytestrings import _BytestringT
from vyper.semantics.types.shortcuts import BYTES32_T
from vyper.utils import SHA3_BASE, SHA3_PER_WORD, MemoryPositions, bytes_to_int, keccak256


def _check_byteslike(typ, _expr):
    if not isinstance(typ, _BytestringT) and typ != BYTES32_T:
        # NOTE this may be checked at a higher level, but just be safe
        raise CompilerPanic("keccak256 only accepts bytes-like objects")


def _gas_bound(num_words):
    return SHA3_BASE + num_words * SHA3_PER_WORD


def keccak256_helper(expr, to_hash, context):
    _check_byteslike(to_hash.typ, expr)

    # Can hash literals
    # TODO this is dead code.
    if isinstance(to_hash, bytes):
        return IRnode.from_list(bytes_to_int(keccak256(to_hash)), typ=BYTES32_T)

    # Can hash bytes32 objects
    # TODO: Want to generalize to all bytes_M
    if to_hash.typ == BYTES32_T:
        return IRnode.from_list(
            [
                "seq",
                ["mstore", MemoryPositions.FREE_VAR_SPACE, to_hash],
                ["sha3", MemoryPositions.FREE_VAR_SPACE, 32],
            ],
            typ=BYTES32_T,
            add_gas_estimate=_gas_bound(1),
        )

    to_hash = ensure_in_memory(to_hash, context)

    with to_hash.cache_when_complex("buf") as (b1, to_hash):
        data = bytes_data_ptr(to_hash)
        len_ = get_bytearray_length(to_hash)
        return b1.resolve(
            IRnode.from_list(
                ["sha3", data, len_],
                typ=BYTES32_T,
                annotation="keccak256",
                add_gas_estimate=_gas_bound(ceil(to_hash.typ.maxlen / 32)),
            )
        )
