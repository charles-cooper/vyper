# unit tests for effects

from vyper.evm.effects import EVMEffects, can_elide
from vyper.codegen.ir_node import IRnode

EFFECT_FREE = EVMEffects.EffectFree
READ = EVMEffects.StorageRead
WRITE = EVMEffects.StorageWrite
STATIC_CALL = EVMEffects.StaticCall
CALL = EVMEffects.Call

def test_can_elide():
    assert can_elide(EFFECT_FREE)
    assert can_elide(READ)
    assert can_elide(STATIC_CALL)
    assert can_elide(READ | STATIC_CALL)
    assert can_elide(EFFECT_FREE | READ | STATIC_CALL)
    assert not can_elide(WRITE)
    assert not can_elide(CALL)
    assert not can_elide(WRITE | READ)
    assert not can_elide(WRITE | STATIC_CALL)
    assert not can_elide(WRITE | EFFECT_FREE)
    assert not can_elide(WRITE | READ | STATIC_CALL | EFFECT_FREE)
    assert not can_elide(CALL | READ)
    assert not can_elide(CALL | STATIC_CALL)
    assert not can_elide(CALL | EFFECT_FREE)
    assert not can_elide(CALL | READ | STATIC_CALL | EFFECT_FREE)
    assert not can_elide(CALL | WRITE)
    assert not can_elide(CALL | WRITE | READ | STATIC_CALL | EFFECT_FREE)


def test_ir_node_effects():
    simple_add = IRnode.from_list(["add", "x", "y"])
    simple_sstore = IRnode.from_list(["sstore", 0, 0])
    simple_sload = IRnode.from_list(["sload", 0])
    simple_call = IRnode.from_list(["call", "gas", 0, "selfbalance", 0, 0, 0, 0])
    simple_static_call = IRnode.from_list(["staticcall", "gas", 0, 0, 0, 0, 0])

    complex_add = IRnode.from_list(["add", simple_sload, simple_call])
    complex_seq = IRnode.from_list(["seq", simple_add, simple_sstore, simple_sload, simple_call, simple_static_call])

    assert simple_add.effects == EFFECT_FREE
    assert simple_sload.effects == READ
    assert simple_static_call.effects == STATIC_CALL
    assert simple_sstore.effects == WRITE
    assert simple_call.effects == CALL
    assert complex_add.effects == CALL | READ
    assert complex_seq.effects == READ | WRITE | STATIC_CALL | CALL
