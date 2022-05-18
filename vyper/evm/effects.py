import enum

class EVMEffects(enum.Flag):
    EffectFree = 0
    StorageRead = enum.auto()
    StaticCall = enum.auto()
    StorageWrite = enum.auto()
    Call = enum.auto()


side_effecting_ops = EVMEffects.StorageWrite | EVMEffects.Call

# for EVMEffects on a given IRnode, can we drop this IRnode?
def can_elide(effects: EVMEffects) -> bool:
    return not (effects & side_effecting_ops)
