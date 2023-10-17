from vyper.codegen.ir_basicblock import IRBasicBlock
from vyper.codegen.ir_function import IRFunction
from vyper.utils import OrderedSet


def _process_basic_block(ctx: IRFunction, bb: IRBasicBlock):
    pass


def ir_pass_constant_propagation(ctx: IRFunction):
    global visited_instructions
    visited_instructions = OrderedSet()

    for bb in ctx.basic_blocks:
        _process_basic_block(ctx, bb)
