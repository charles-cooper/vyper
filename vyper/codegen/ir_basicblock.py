from enum import Enum
from typing import TYPE_CHECKING, Optional

from vyper.utils import OrderedSet

# instructions which can terminate a basic block
BB_TERMINATORS = ["jmp", "jnz", "ret", "return", "revert", "deploy", "stop"]

if TYPE_CHECKING:
    from vyper.codegen.ir_function import IRFunction


class IRDebugInfo:
    """
    IRDebugInfo represents debug information in IR, used to annotate IR instructions
    with source code information when printing IR.
    """

    line_no: int
    src: str

    def __init__(self, line_no: int, src: str) -> None:
        self.line_no = line_no
        self.src = src

    def __repr__(self) -> str:
        src = self.src if self.src else ""
        return f"\t# line {self.line_no}: {src}".expandtabs(20)


IRValueBaseValue = str | int


class IRValueBase:
    value: IRValueBaseValue

    def __init__(self, value: IRValueBaseValue) -> None:
        assert isinstance(value, IRValueBaseValue), "value must be an IRValueBaseValue"
        self.value = value

    @property
    def is_literal(self) -> bool:
        return False

    def __repr__(self) -> str:
        return str(self.value)


class IRLiteral(IRValueBase):
    """
    IRLiteral represents a literal in IR
    """

    def __init__(self, value: IRValueBaseValue) -> None:
        super().__init__(value)

    @property
    def is_literal(self) -> bool:
        return True


class IRVariable(IRValueBase):
    """
    IRVariable represents a variable in IR. A variable is a string that starts with a %.
    """

    offset: int = 0
    # REVIEW: make this toplevel definition
    MemType = Enum("MemType", ["OPERAND_STACK", "MEMORY"])
    mem_type: MemType = MemType.OPERAND_STACK
    mem_addr: int = -1  # REVIEW should this be None?

    def __init__(
        self, value: IRValueBaseValue, mem_type: MemType = MemType.OPERAND_STACK, mem_addr: int = -1
    ) -> None:
        if isinstance(value, IRLiteral):
            value = value.value
        super().__init__(value)
        self.offset = 0
        self.mem_type = mem_type
        self.mem_addr = mem_addr


class IRLabel(IRValueBase):
    # REVIEW: what do the values of is_symbol mean?
    """
    IRLabel represents a label in IR. A label is a string that starts with a %.
    """

    is_symbol: bool = False

    def __init__(self, value: str, is_symbol: bool = False) -> None:
        super().__init__(value)
        self.is_symbol = is_symbol


class IRInstruction:
    """
    IRInstruction represents an instruction in IR. Each instruction has an opcode,
    operands, and return value. For example, the following IR instruction:
        %1 = add %0, 1
    has opcode "add", operands ["%0", "1"], and return value "%1".
    """

    opcode: str
    volatile: bool
    operands: list[IRValueBase]
    # REVIEW: rename to lhs?
    ret: Optional[IRValueBase]
    # REVIEW: rename to source_info?
    dbg: Optional[IRDebugInfo]
    # set of live variables at this instruction
    liveness: OrderedSet[IRVariable]
    dup_requirements: OrderedSet[IRVariable]
    parent: Optional["IRBasicBlock"]
    # REVIEW: rename to `fence`
    fen: int
    annotation: Optional[str]

    def __init__(
        self,
        opcode: str,
        operands: list[IRValueBase],
        ret: IRValueBase = None,
        dbg: IRDebugInfo = None,
    ):
        self.opcode = opcode
        # REVIEW nit: make this global definition
        self.volatile = opcode in [
            "param",
            "alloca",
            "call",
            "staticcall",
            "invoke",
            "sload",
            "sstore",
            "iload",
            "istore",
            "assert",
            "mstore",
            "mload",
            "calldatacopy",
            "codecopy",
            "dloadbytes",
            "dload",
            "return",
            "ret",
            "jmp",
            "jnz",
        ]
        self.operands = [op if isinstance(op, IRValueBase) else IRValueBase(op) for op in operands]
        self.ret = ret if isinstance(ret, IRValueBase) else IRValueBase(ret) if ret else None
        self.dbg = dbg
        self.liveness = OrderedSet()
        self.dup_requirements = OrderedSet()
        self.parent = None
        self.fen = -1
        self.annotation = None

    def get_label_operands(self) -> list[IRLabel]:
        """
        Get all labels in instruction.
        """
        return [op for op in self.operands if isinstance(op, IRLabel)]

    def get_non_label_operands(self) -> list[IRValueBase]:
        """
        Get all input operands in instruction.
        """
        return [op for op in self.operands if not isinstance(op, IRLabel)]

    def get_input_operands(self) -> list[IRValueBase]:
        """
        Get all input operands for instruction.
        """
        return [op for op in self.operands if isinstance(op, IRVariable)]

    # REVIEW suggestion: rename to `get_outputs`
    def get_output_operands(self) -> list[IRValueBase]:
        return [self.ret] if self.ret else []

    # REVIEW: rename to `replace_operands`
    # use of `dict` here seems a bit weird (what is equality on operands?)
    def update_operands(self, replacements: dict) -> None:
        """
        Update operands with replacements.
        """
        for i, operand in enumerate(self.operands):
            if operand in replacements.keys():
                self.operands[i] = replacements[operand]

    def __repr__(self) -> str:
        s = ""
        if self.ret:
            s += f"{self.ret} = "
        opcode = f"{self.opcode} " if self.opcode != "store" else ""
        s += opcode
        operands = ", ".join(
            [(f"label %{op}" if isinstance(op, IRLabel) else str(op)) for op in self.operands[::-1]]
        )
        s += operands

        if self.dbg:
            return s + f" {self.dbg}"

        if self.annotation:
            s += f" <{self.annotation}>"

        if self.liveness:
            return f"{s: <30} # {self.liveness}"

        return s


class IRBasicBlock:
    """
    IRBasicBlock represents a basic block in IR. Each basic block has a label and
    a list of instructions, while belonging to a function.

    The following IR code:
        %1 = add %0, 1
        %2 = mul %1, 2
    is represented as:
        bb = IRBasicBlock("bb", function)
        bb.append_instruction(IRInstruction("add", ["%0", "1"], "%1"))
        bb.append_instruction(IRInstruction("mul", ["%1", "2"], "%2"))

    The label of a basic block is used to refer to it from other basic blocks
    in order to branch to it.

    The parent of a basic block is the function it belongs to.

    The instructions of a basic block are executed sequentially, and the last
    instruction of a basic block is always a terminator instruction, which is
    used to branch to other basic blocks.
    """

    label: IRLabel
    parent: "IRFunction"
    instructions: list[IRInstruction]
    # REVIEW: "in_set" -> "cfg_in"
    # (basic blocks which can jump to this basic block)
    cfg_in: OrderedSet["IRBasicBlock"]
    # REVIEW: "out_set" -> "cfg_out"
    # (basic blocks which this basic block can jump to)
    cfg_out: OrderedSet["IRBasicBlock"]
    # stack items which this basic block produces
    out_vars: OrderedSet[IRVariable]

    def __init__(self, label: IRLabel, parent: "IRFunction") -> None:
        assert isinstance(label, IRLabel), "label must be an IRLabel"
        self.label = label
        self.parent = parent
        self.instructions = []
        self.cfg_in = OrderedSet()
        self.cfg_out = OrderedSet()
        self.out_vars = OrderedSet()

    def add_cfg_in(self, bb: "IRBasicBlock") -> None:
        self.cfg_in.add(bb)

    def union_cfg_in(self, bb_set: OrderedSet["IRBasicBlock"]) -> None:
        self.cfg_in = self.cfg_in.union(bb_set)

    def remove_cfg_in(self, bb: "IRBasicBlock") -> None:
        self.cfg_in.remove(bb)

    def add_cfg_out(self, bb: "IRBasicBlock") -> None:
        self.cfg_out.add(bb)

    def union_cfg_out(self, bb_set: OrderedSet["IRBasicBlock"]) -> None:
        self.cfg_out = self.cfg_out.union(bb_set)

    def remove_cfg_out(self, bb: "IRBasicBlock") -> None:
        self.cfg_out.remove(bb)

    # calculate the input variables into self from source
    def in_vars_from(self, source: "IRBasicBlock") -> OrderedSet[IRVariable]:
        liveness = self.instructions[0].liveness.copy()
        assert isinstance(liveness, OrderedSet)

        for inst in self.instructions:
            # REVIEW: might be nice if some of these instructions
            # were more structured.
            if inst.opcode == "select":
                if inst.operands[0] == source.label:
                    liveness.add(inst.operands[1])
                    if inst.operands[3] in liveness:
                        liveness.remove(inst.operands[3])
                if inst.operands[2] == source.label:
                    liveness.add(inst.operands[3])
                    if inst.operands[1] in liveness:
                        liveness.remove(inst.operands[1])

        return liveness

    @property
    def is_reachable(self) -> bool:
        return len(self.cfg_in) > 0

    def append_instruction(self, instruction: IRInstruction) -> None:
        assert isinstance(instruction, IRInstruction), "instruction must be an IRInstruction"
        instruction.parent = self
        self.instructions.append(instruction)

    def insert_instruction(self, instruction: IRInstruction, index: int) -> None:
        assert isinstance(instruction, IRInstruction), "instruction must be an IRInstruction"
        instruction.parent = self
        self.instructions.insert(index, instruction)

    def clear_instructions(self) -> None:
        self.instructions = []

    # REVIEW: rename to replace_operands
    def update_operands(self, replacements: dict) -> None:
        """
        Update operands with replacements.
        """
        for instruction in self.instructions:
            instruction.update_operands(replacements)

    @property
    def is_terminated(self) -> bool:
        """
        Check if the basic block is terminal, i.e. the last instruction is a terminator.
        """
        # REVIEW: should this be an assert (like `is_terminal()`)?
        if len(self.instructions) == 0:
            return False
        return self.instructions[-1].opcode in BB_TERMINATORS

    def calculate_liveness(self) -> None:
        """
        Compute liveness of each instruction in the basic block.
        """
        liveness = self.out_vars.copy()
        # REVIEW: use `reversed()` here
        for instruction in self.instructions[::-1]:
            ops = instruction.get_input_operands()
            liveness = liveness.union(OrderedSet.fromkeys(ops))
            out = (
                instruction.get_output_operands()[0]
                if len(instruction.get_output_operands()) > 0
                else None
            )
            if out in liveness:
                liveness.remove(out)
            instruction.liveness = liveness

    def copy(self):
        bb = IRBasicBlock(self.label, self.parent)
        bb.instructions = self.instructions.copy()
        bb.cfg_in = self.cfg_in.copy()
        bb.cfg_out = self.cfg_out.copy()
        bb.out_vars = self.out_vars.copy()
        return bb

    def __repr__(self) -> str:
        s = (
            f"{repr(self.label)}:  IN={[bb.label for bb in self.cfg_in]}"
            f" OUT={[bb.label for bb in self.cfg_out]} \n"
        )
        for instruction in self.instructions:
            s += f"    {instruction}\n"
        return s
