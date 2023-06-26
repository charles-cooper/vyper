from typing import Optional, Union
from vyper.codegen.global_context import GlobalContext
from vyper.codegen.ir_node import IRnode
from vyper.codegen.ir_function import IRFunction, IRFunctionIntrinsic
from vyper.codegen.ir_basicblock import IRInstruction, IRDebugInfo
from vyper.codegen.ir_basicblock import IRBasicBlock
from vyper.evm.opcodes import get_opcodes

_symbols = {}


def convert_ir_basicblock(ctx: GlobalContext, ir: IRnode) -> IRFunction:
    global_function = IRFunction("global")
    _convert_ir_basicblock(global_function, ir)
    return global_function


def _convert_binary_op(ctx: IRFunction, ir: IRnode) -> str:
    arg_0 = _convert_ir_basicblock(ctx, ir.args[0])
    arg_1 = _convert_ir_basicblock(ctx, ir.args[1])
    args = [arg_0, arg_1]

    ret = ctx.get_next_variable()

    inst = IRInstruction(str(ir.value), args, ret)
    ctx.get_basic_block().append_instruction(inst)
    return ret


def _convert_ir_basicblock(ctx: IRFunction, ir: IRnode) -> Optional[Union[str, int]]:
    if ir.value == "deploy":
        _convert_ir_basicblock(ctx, ir.args[1])
    elif ir.value == "seq":
        for ir_node in ir.args:
            _convert_ir_basicblock(ctx, ir_node)
    elif ir.value == "if":
        cond = ir.args[0]
        _convert_ir_basicblock(ctx, cond)
        _convert_ir_basicblock(ctx, ir.args[1])
    elif ir.value == "with":
        ret = _convert_ir_basicblock(ctx, ir.args[1])  # initialization

        sym = ir.args[0]
        # FIXME: How do I validate that the IR is indeed a symbol?
        _symbols[sym.value] = ctx.get_next_variable()
        first_pos = ir.source_pos[0] if ir.source_pos else None
        inst = IRInstruction(
            "load",
            [_symbols[sym.value], ret],
            None,
            IRDebugInfo(first_pos or 0, f"symbol: {sym.value}"),
        )
        ctx.get_basic_block().append_instruction(inst)

        _convert_ir_basicblock(ctx, ir.args[2])  # body
    elif ir.value in ["le", "ge", "shr", "xor"]:
        return _convert_binary_op(ctx, ir)
    elif ir.value == "iszero":
        arg_0 = _convert_ir_basicblock(ctx, ir.args[0])
        args = [arg_0]

        ret = ctx.get_next_variable()

        inst = IRInstruction("iszero", args, ret)
        ctx.get_basic_block().append_instruction(inst)
        return ret
    elif ir.value == "goto":
        inst = IRInstruction("br", ir.args)
        ctx.get_basic_block().append_instruction(inst)

        label = ctx.get_next_label()
        bb = IRBasicBlock(label, ctx)
        ctx.append_basic_block(bb)
    elif ir.value == "calldatasize":
        ret = ctx.get_next_variable()
        func = IRFunctionIntrinsic("calldatasize", [])
        inst = IRInstruction("call", [func], ret)
        ctx.get_basic_block().append_instruction(inst)
        return ret
    elif ir.value == "calldataload":
        ret = ctx.get_next_variable()
        func = IRFunctionIntrinsic("calldataload", [ir.args[0]])
        inst = IRInstruction("call", [func], ret)
        ctx.get_basic_block().append_instruction(inst)
        return ret
    elif ir.value == "callvalue":
        ret = ctx.get_next_variable()
        func = IRFunctionIntrinsic("callvalue", [])
        inst = IRInstruction("call", [func], ret)
        ctx.get_basic_block().append_instruction(inst)
        return ret
    elif ir.value == "assert":
        arg_0 = _convert_ir_basicblock(ctx, ir.args[0])

        ret = ctx.get_next_variable()
        func = IRFunctionIntrinsic("assert", [arg_0])
        inst = IRInstruction("call", [func], ret)
        ctx.get_basic_block().append_instruction(inst)
        return ret
    elif ir.value == "label":
        label = str(ir.args[0].value)
        bb = IRBasicBlock(label, ctx)
        ctx.append_basic_block(bb)
        _convert_ir_basicblock(ctx, ir.args[2])
    elif ir.value == "return":
        pass
    elif ir.value == "exit_to":
        pass
    elif ir.value == "pass":
        pass
    elif ir.value == "mstore":
        sym = ir.args[0]
        new_var = ctx.get_next_variable()
        _symbols[f"&{sym.value}"] = new_var
        assert ir.args[1].is_literal, "mstore expects a literal as second argument"
        first_pos = ir.source_pos[0] if ir.source_pos else None
        inst = IRInstruction(
            "load",
            [new_var, ir.args[1].value],
            None,
            IRDebugInfo(first_pos or 0, ir.annotation or ""),
        )
        ctx.get_basic_block().append_instruction(inst)
    elif isinstance(ir.value, str) and ir.value.upper() in get_opcodes():
        _convert_ir_opcode(ctx, ir)
    elif isinstance(ir.value, str) and ir.value in _symbols:
        return _symbols[ir.value]
    elif ir.is_literal:
        return ir.value
    else:
        raise Exception(f"Unknown IR node: {ir}")

    return None


def _convert_ir_opcode(ctx: IRFunction, ir: IRnode) -> None:
    opcode = str(ir.value).upper()
    for arg in ir.args:
        if isinstance(arg, IRnode):
            _convert_ir_basicblock(ctx, arg)
    instruction = IRInstruction(opcode, ir.args)
    ctx.get_basic_block().append_instruction(instruction)
