# Module for codegen. Currently most codegen lives in
# parser/parser_utils.py and can slowly be migrated here as
# type-checking code gets factored out.

from vyper.exceptions import (
    InvalidLiteralException,
    TypeMismatchException,
    StructureException
)
from vyper.types import (
    BaseType,
    ByteArrayType,
    ContractType,
    NullType,
    StructType,
    MappingType,
    TupleType,
    TupleLike,
    ListType,
    get_size_of_type,
)
from vyper.parser.parser_utils import (
    base_type_conversion,
    getpos,
    LLLnode,
    make_byte_array_copier,
    make_setter,
    unwrap_location,
    add_variable_offset,
)


# Generate return code for stmt
def make_return_stmt(stmt, context, begin_pos, _size, loop_memory_position=None):
    if context.is_private:
        if loop_memory_position is None:
            loop_memory_position = context.new_placeholder(typ=BaseType('uint256'))

        # Make label for stack push loop.
        label_id = '_'.join([str(x) for x in (context.method_id, stmt.lineno, stmt.col_offset)])
        exit_label = 'make_return_loop_exit_%s' % label_id
        start_label = 'make_return_loop_start_%s' % label_id

        # Push prepared data onto the stack,
        # in reverse order so it can be popped of in order.
        if _size == 0:
            mloads = []
        elif isinstance(begin_pos, int) and isinstance(_size, int):
            # static values, unroll the mloads instead.
            mloads = [
                ['mload', pos] for pos in range(begin_pos, _size, 32)
            ]
            return ['seq_unchecked'] + mloads + [['jump', ['mload', context.callback_ptr]]]
        else:
            mloads = [
                'seq_unchecked',
                ['mstore', loop_memory_position, _size],
                ['label', start_label],
                ['if',
                    ['le', ['mload', loop_memory_position], 0], ['goto', exit_label]],  # exit loop / break.
                ['mload', ['add', begin_pos, ['sub', ['mload', loop_memory_position], 32]]],  # push onto stack
                ['mstore', loop_memory_position, ['sub', ['mload', loop_memory_position], 32]],  # decrement i by 32.
                ['goto', start_label],
                ['label', exit_label]
            ]
            return ['seq_unchecked'] + [mloads] + [['jump', ['mload', context.callback_ptr]]]
    else:
        return ['return', begin_pos, _size]

def make_return_stmt_multi(stmt, context, sub):
    lmp = context.new_placeholder(typ=BaseType('uint256')) # Loop memory position
    typ = context.return_type
    len_ = get_size_of_type(typ) * 32

    if sub.location == "memory":
        r = make_return_stmt(stmt, context, sub, len_, loop_memory_position=lmp)
        return LLLnode.from_list(r, typ=None, pos=getpos(stmt), valency=0)

    else:
        x = context.new_placeholder(typ)
        new_sub = LLLnode.from_list(x, typ=typ, location='memory')
        setter = make_setter(new_sub, sub, 'memory', pos=getpos(stmt))
        r = make_return_stmt(stmt, context, new_sub, len_, loop_memory_position=lmp)
        return LLLnode.from_list(
                ['seq',
                    setter,
                    r],
                typ=None,
                pos=getpos(stmt))
