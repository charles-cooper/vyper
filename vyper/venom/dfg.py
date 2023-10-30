from vyper.venom.basicblock import (
    IRBasicBlock,
    IRInstruction,
    IRLabel,
    IRValueBase,
    IRVariable,
    MemType,
)
from vyper.venom.function import IRFunction
from vyper.venom.stack_model import StackModel
from vyper.ir.compile_ir import PUSH, DataHeader, RuntimeHeader, optimize_assembly
from vyper.utils import MemoryPositions, OrderedSet

ONE_TO_ONE_INSTRUCTIONS = [
    "revert",
    "coinbase",
    "calldatasize",
    "calldatacopy",
    "calldataload",
    "gas",
    "gasprice",
    "gaslimit",
    "address",
    "origin",
    "number",
    "extcodesize",
    "extcodehash",
    "returndatasize",
    "returndatacopy",
    "callvalue",
    "selfbalance",
    "sload",
    "sstore",
    "mload",
    "mstore",
    "timestamp",
    "caller",
    "selfdestruct",
    "signextend",
    "stop",
    "shr",
    "shl",
    "and",
    "xor",
    "or",
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "exp",
    "eq",
    "iszero",
    "lg",
    "lt",
    "slt",
    "sgt",
    "log0",
    "log1",
    "log2",
    "log3",
    "log4",
]


class DFGNode:
    value: IRInstruction | IRValueBase
    predecessors: list["DFGNode"]
    successors: list["DFGNode"]

    def __init__(self, value: IRInstruction | IRValueBase):
        self.value = value
        self.predecessors = []
        self.successors = []


def convert_ir_to_dfg(ctx: IRFunction) -> None:
    # Reset DFG
    # REVIEW: dfg inputs is all, flattened inputs to a given variable
    ctx.dfg_inputs = dict()
    # REVIEW: dfg outputs is the instruction which produces a variable
    ctx.dfg_outputs = dict()
    for bb in ctx.basic_blocks:
        for inst in bb.instructions:
            inst.dup_requirements = OrderedSet()
            inst.fence_id = -1
            operands = inst.get_inputs()
            operands.extend(inst.get_outputs())

    # Build DFG
    for bb in ctx.basic_blocks:
        for inst in bb.instructions:
            operands = inst.get_inputs()
            res = inst.get_outputs()

            for op in operands:
                ctx.dfg_inputs[op.value] = (
                    [inst]
                    if ctx.dfg_inputs.get(op.value) is None
                    else ctx.dfg_inputs[op.value] + [inst]
                )

            for op in res:
                ctx.dfg_outputs[op.value] = inst

    # Build DUP requirements
    _compute_dup_requirements(ctx)


def _compute_inst_dup_requirements_r(
    ctx: IRFunction,
    inst: IRInstruction,
    visited: OrderedSet,
    last_seen: dict,
) -> None:
    for op in inst.get_outputs():
        for target in ctx.dfg_inputs.get(op.value, []):
            if target.parent != inst.parent:
                # REVIEW: produced by parent.out_vars
                continue
            if target.fence_id != inst.fence_id:
                continue
            _compute_inst_dup_requirements_r(ctx, target, visited, last_seen)

    if inst in visited:
        return
    visited.add(inst)

    if inst.opcode == "phi":
        return

    for op in inst.get_inputs():
        target = ctx.dfg_outputs[op.value]
        if target.parent != inst.parent:
            continue
        _compute_inst_dup_requirements_r(ctx, target, visited, last_seen)

    for op in inst.get_inputs():
        target = last_seen.get(op.value, None)
        if target:
            target.dup_requirements.add(op)
        last_seen[op.value] = inst

    return


def _compute_dup_requirements(ctx: IRFunction) -> None:
    fen = 0
    for bb in ctx.basic_blocks:
        for inst in bb.instructions:
            inst.fence_id = fen
            if inst.volatile:
                fen += 1

        visited = OrderedSet()
        last_seen = dict()
        for inst in bb.instructions:
            _compute_inst_dup_requirements_r(ctx, inst, visited, last_seen)

        out_vars = bb.out_vars
        for inst in reversed(bb.instructions):
            for op in inst.get_inputs():
                if op in out_vars:
                    inst.dup_requirements.add(op)


visited_instructions = None  # {IRInstruction}
visited_basicblocks = None  # {IRBasicBlock}


def generate_evm(ctx: IRFunction, no_optimize: bool = False) -> list[str]:
    global visited_instructions, visited_basicblocks
    asm = []
    visited_instructions = OrderedSet()
    visited_basicblocks = OrderedSet()

    _generate_evm_for_basicblock_r(ctx, asm, ctx.basic_blocks[0], StackModel())

    # Append postambles
    revert_postamble = ["_sym___revert", "JUMPDEST", *PUSH(0), "DUP1", "REVERT"]
    runtime = None
    if isinstance(asm[-1], list) and isinstance(asm[-1][0], RuntimeHeader):
        runtime = asm.pop()

    asm.extend(revert_postamble)
    if runtime:
        runtime.extend(revert_postamble)
        asm.append(runtime)

    # Append data segment
    data_segments = {}
    for inst in ctx.data_segment:
        if inst.opcode == "dbname":
            label = inst.operands[0].value
            data_segments[label] = [DataHeader(f"_sym_{label}")]
        elif inst.opcode == "db":
            data_segments[label].append(f"_sym_{inst.operands[0].value}")

    extent_point = asm if not isinstance(asm[-1], list) else asm[-1]
    extent_point.extend([data_segments[label] for label in data_segments])

    if no_optimize is False:
        optimize_assembly(asm)

    return asm


def _stack_duplications(
    assembly: list, inst: IRInstruction, stack_map: StackModel, stack_ops: list[IRValueBase]
) -> None:
    for op in stack_ops:
        if op.is_literal or isinstance(op, IRLabel):
            continue
        depth = stack_map.get_depth_in(op)
        assert depth is not StackModel.NOT_IN_STACK, "Operand not in stack"
        if op in inst.dup_requirements:
            stack_map.dup(assembly, depth)


def _stack_reorder(assembly: list, stack_map: StackModel, stack_ops: list[IRValueBase]) -> None:
    stack_ops = [x.value for x in stack_ops]
    # print("ENTER reorder", stack_map.stack_map, operands)
    # start_len = len(assembly)
    for i in range(len(stack_ops)):
        op = stack_ops[i]
        final_stack_depth = -(len(stack_ops) - i - 1)
        depth = stack_map.get_depth_in(op)
        assert depth is not StackModel.NOT_IN_STACK, f"{op} not in stack: {stack_map.stack}"
        if depth == final_stack_depth:
            continue

        # print("trace", depth, final_stack_depth)
        stack_map.swap(assembly, depth)
        stack_map.swap(assembly, final_stack_depth)

    # print("INSTRUCTIONS", assembly[start_len:])
    # print("EXIT reorder", stack_map.stack_map, stack_ops)


def _generate_evm_for_basicblock_r(
    ctx: IRFunction, asm: list, basicblock: IRBasicBlock, stack_map: StackModel
):
    if basicblock in visited_basicblocks:
        return
    visited_basicblocks.add(basicblock)

    asm.append(f"_sym_{basicblock.label}")
    asm.append("JUMPDEST")

    fen = 0
    for inst in basicblock.instructions:
        inst.fence_id = fen
        if inst.volatile:
            fen += 1

    for inst in basicblock.instructions:
        asm = _generate_evm_for_instruction_r(ctx, asm, inst, stack_map)

    for bb in basicblock.cfg_out:
        _generate_evm_for_basicblock_r(ctx, asm, bb, stack_map.copy())


# TODO: refactor this
label_counter = 0


# REVIEW: would this be better as a class?
def _generate_evm_for_instruction_r(
    ctx: IRFunction, assembly: list, inst: IRInstruction, stack_map: StackModel
) -> list[str]:
    global label_counter

    for op in inst.get_outputs():
        for target in ctx.dfg_inputs.get(op.value, []):
            # REVIEW: what does this line do?
            # HK: it skips instructions that are not in the same basic block
            #     so we don't cross basic block boundaries
            if target.parent != inst.parent:
                continue
            # REVIEW: what does this line do?
            # HK: it skips instructions that are not in the same fence
            if target.fence_id != inst.fence_id:
                continue
            # REVIEW: I think it would be better to have an explicit step,
            # `reorder instructions per DFG`, and then `generate_evm_for_instruction`
            # does not need to recurse (or be co-recursive with `emit_input_operands`).
            # HK: Indeed, this is eventualy the idea. Especialy now that I have implemented
            #     the "needs duplication" algorithm that needs the same traversal and it's duplicated
            assembly = _generate_evm_for_instruction_r(ctx, assembly, target, stack_map)

    if inst in visited_instructions:
        # print("seen:", inst)
        return assembly
    visited_instructions.add(inst)

    opcode = inst.opcode

    #
    # generate EVM for op
    #

    # Step 1: Apply instruction special stack manipulations

    if opcode in ["jmp", "jnz", "invoke"]:
        operands = inst.get_non_label_operands()
    elif opcode == "alloca":
        operands = inst.operands[1:2]
    elif opcode == "iload":
        operands = []
    elif opcode == "istore":
        operands = inst.operands[0:1]
    else:
        operands = inst.operands

    if opcode == "phi":
        ret = inst.get_outputs()[0]
        inputs = inst.get_inputs()
        # REVIEW: the special handling in get_depth_in for lists
        # seems cursed, refactor
        depth = stack_map.get_depth_in(inputs)
        assert depth is not StackModel.NOT_IN_STACK, "Operand not in stack"
        to_be_replaced = stack_map.peek(depth)
        if to_be_replaced in inst.dup_requirements:
            stack_map.dup(assembly, depth)
            stack_map.poke(0, ret)
        else:
            stack_map.poke(depth, ret)
        return assembly

    # Step 2: Emit instruction's input operands
    _emit_input_operands(ctx, assembly, inst, operands, stack_map)

    # Step 3: Reorder stack
    if opcode in ["jnz", "jmp"]:
        assert isinstance(inst.parent.cfg_out, OrderedSet)
        b = next(iter(inst.parent.cfg_out))
        target_stack = OrderedSet(b.in_vars_from(inst.parent))
        _stack_reorder(assembly, stack_map, target_stack)

    _stack_duplications(assembly, inst, stack_map, operands)

    # print("(inst)", inst)
    _stack_reorder(assembly, stack_map, operands)

    # Step 4: Push instruction's return value to stack
    stack_map.pop(len(operands))
    if inst.ret is not None:
        stack_map.push(inst.ret)

    # Step 5: Emit the EVM instruction(s)
    if opcode in ONE_TO_ONE_INSTRUCTIONS:
        assembly.append(opcode.upper())
    elif opcode == "alloca":
        pass
    elif opcode == "param":
        pass
    elif opcode == "store":
        pass
    elif opcode == "dbname":
        pass
    elif opcode in ["codecopy", "dloadbytes"]:
        assembly.append("CODECOPY")
    elif opcode == "jnz":
        assembly.append(f"_sym_{inst.operands[1].value}")
        assembly.append("JUMPI")
    elif opcode == "jmp":
        if isinstance(inst.operands[0], IRLabel):
            assembly.append(f"_sym_{inst.operands[0].value}")
            assembly.append("JUMP")
        else:
            assembly.append("JUMP")
    elif opcode == "gt":
        assembly.append("GT")
    elif opcode == "lt":
        assembly.append("LT")
    elif opcode == "invoke":
        target = inst.operands[0]
        assert isinstance(target, IRLabel), "invoke target must be a label"
        assembly.extend(
            [
                f"_sym_label_ret_{label_counter}",
                f"_sym_{target.value}",
                "JUMP",
                f"_sym_label_ret_{label_counter}",
                "JUMPDEST",
            ]
        )
        label_counter += 1
        if stack_map.get_height() > 0 and stack_map.peek(0) in inst.dup_requirements:
            stack_map.pop()
            assembly.append("POP")
    elif opcode == "call":
        assembly.append("CALL")
    elif opcode == "staticcall":
        assembly.append("STATICCALL")
    elif opcode == "ret":
        # assert len(inst.operands) == 2, "ret instruction takes two operands"
        assembly.append("JUMP")
    elif opcode == "return":
        assembly.append("RETURN")
    elif opcode == "phi":
        pass
    elif opcode == "sha3":
        assembly.append("SHA3")
    elif opcode == "sha3_64":
        assembly.extend(
            [
                *PUSH(MemoryPositions.FREE_VAR_SPACE2),
                "MSTORE",
                *PUSH(MemoryPositions.FREE_VAR_SPACE),
                "MSTORE",
                *PUSH(64),
                *PUSH(MemoryPositions.FREE_VAR_SPACE),
                "SHA3",
            ]
        )
    elif opcode == "ceil32":
        assembly.extend([*PUSH(31), "ADD", *PUSH(31), "NOT", "AND"])
    elif opcode == "assert":
        assembly.extend(["ISZERO", "_sym___revert", "JUMPI"])
    elif opcode == "deploy":
        memsize = inst.operands[0].value
        padding = inst.operands[2].value
        # TODO: fix this by removing deploy opcode altogether me move emition to ir translation
        while assembly[-1] != "JUMPDEST":
            assembly.pop()
        assembly.extend(
            ["_sym_subcode_size", "_sym_runtime_begin", "_mem_deploy_start", "CODECOPY"]
        )
        assembly.extend(["_OFST", "_sym_subcode_size", padding])  # stack: len
        assembly.extend(["_mem_deploy_start"])  # stack: len mem_ofst
        assembly.extend(["RETURN"])
        assembly.append([RuntimeHeader("_sym_runtime_begin", memsize, padding)])
        assembly = assembly[-1]
    elif opcode == "iload":
        loc = inst.operands[0].value
        assembly.extend(["_OFST", "_mem_deploy_end", loc, "MLOAD"])
    elif opcode == "istore":
        loc = inst.operands[1].value
        assembly.extend(["_OFST", "_mem_deploy_end", loc, "MSTORE"])
    else:
        raise Exception(f"Unknown opcode: {opcode}")

    # Step 6: Emit instructions output operands (if any)
    if inst.ret is not None:
        assert isinstance(inst.ret, IRVariable), "Return value must be a variable"
        if inst.ret.mem_type == MemType.MEMORY:
            assembly.extend([*PUSH(inst.ret.mem_addr)])

    return assembly


def _emit_input_operands(
    ctx: IRFunction,
    assembly: list,
    inst: IRInstruction,
    ops: list[IRValueBase],
    stack_map: StackModel,
):
    # print("EMIT INPUTS FOR", inst)
    for op in ops:
        if isinstance(op, IRLabel):
            # invoke emits the actual instruction itself so we don't need to emit it here
            # but we need to add it to the stack map
            if inst.opcode != "invoke":
                assembly.append(f"_sym_{op.value}")
            stack_map.push(op)
            continue
        if op.is_literal:
            assembly.extend([*PUSH(op.value)])
            stack_map.push(op)
            continue
        # print("RECURSE FOR", op, "TO:", ctx.dfg_outputs[op.value])
        assembly.extend(
            _generate_evm_for_instruction_r(ctx, [], ctx.dfg_outputs[op.value], stack_map)
        )
        if isinstance(op, IRVariable) and op.mem_type == MemType.MEMORY:
            assembly.extend([*PUSH(op.mem_addr)])
            assembly.append("MLOAD")