from vyper.utils import evm_not
from vyper.venom.basicblock import IRBasicBlock, IRInstruction, IRLiteral
from vyper.venom.passes.base_pass import IRPass

# not takes 1 byte1, so it makes sense to use it when we can save at least
# 1 byte
NOT_THRESHOLD = 1

# shl takes 3 bytes, so it makes sense to use it when we can save at least
# 3 bytes
SHL_THRESHOLD = 3


class ReduceLiteralsCodesize(IRPass):
    def run_pass(self):
        for bb in self.function.get_basic_blocks():
            self._process_bb(bb)

    def _process_bb(self, bb: IRBasicBlock):
        i = 0
        while i < len(bb.instructions):
            inst = bb.instructions[i]
            i += 1
            if inst.opcode != "store":
                continue

            op = inst.operands[0]
            if not isinstance(op, IRLiteral):
                continue

            val = op.value % (2**256)

            # calculate amount of bits saved by not optimization
            not_benefit = ((len(hex(val)) // 2 - len(hex(evm_not(val))) // 2) - NOT_THRESHOLD) * 8

            # calculate amount of bits saved by shl optimization
            binz = bin(val)[2:]
            ix = len(binz) - binz.rfind("1")
            shl_benefit = ix - SHL_THRESHOLD * 8

            # calculate amount of bits saved by both optimizations put together
            negated = evm_not(val)
            negated_binz = bin(negated)[2:]
            negated_ix = len(negated_binz) - negated_binz.rfind("1")
            not_then_shl_size = (
                (len(hex(negated)) - 2) * 4 - negated_ix + 1 + NOT_THRESHOLD * 8 + SHL_THRESHOLD * 8
            )
            not_the_shl_benefit = (len(hex(val)) - 2) * 4 - not_then_shl_size

            if not_benefit <= 0 and shl_benefit <= 0 and not_the_shl_benefit <= 0:
                # no optimization can be done here
                continue
            if not_the_shl_benefit > not_benefit and not_the_shl_benefit > shl_benefit:
                negated_ix -= 1
                # sanity check
                assert (negated >> negated_ix) << negated_ix == negated, negated
                assert (negated >> negated_ix) & 1 == 1, negated
                index = bb.instructions.index(inst)
                var = bb.parent.get_next_variable()
                new_inst = IRInstruction(
                    "shl", [IRLiteral(negated >> negated_ix), IRLiteral(negated_ix)], output=var
                )
                bb.insert_instruction(new_inst, index)
                inst.opcode = "not"
                inst.operands = [var]
                continue
            elif not_benefit >= shl_benefit:
                # transform things like 0xffff...01 to (not 0xfe)
                inst.opcode = "not"
                op.value = evm_not(val)
                continue
            else:
                # transform things like 0x123400....000 to 0x1234 << ...
                ix -= 1
                # sanity check
                assert (val >> ix) << ix == val, val
                assert (val >> ix) & 1 == 1, val

                inst.opcode = "shl"
                inst.operands = [IRLiteral(val >> ix), IRLiteral(ix)]
                continue
