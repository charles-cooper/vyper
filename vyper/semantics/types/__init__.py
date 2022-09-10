from . import primitives, subscriptable, user
from .base import TYPE_T, KwargSettings, VyperType
from .bytestrings import BytesT, StringT
from .primitives import AddressT, BoolT, BytesM_T, DecimalT, IntegerT
from .subscriptable import DArrayT, HashMapT, SArrayT, TupleT
from .user import EnumT, EventT, InterfaceT, StructT


def get_primitive_types():
    res = [AddressT(), BoolT(), DecimalT()]

    res.extend(IntegerT.all())
    res.extend(BytesM_T.all())

    # note: since bytestrings are parametrizable, the *class* objects
    # are in the namespace instead of concrete type objects.
    res.extend([BytesT, StringT])

    ret = {t._id: t for t in res}
    ret.update(_get_sequence_types())

    return ret


def _get_sequence_types():
    # since these guys are parametrizable, the *class* objects
    # are in the namespace instead of concrete type objects.

    res = [HashMapT, DArrayT]

    ret = {t._id: t for t in res}

    # (static) arrays and tuples are special types which don't show up
    # in the type annotation itself.
    # since we don't have special handling of annotations in the parser,
    # break a dependency cycle by injecting these into the namespace with
    # mangled names (that no user can create).
    ret["$SArrayT"] = SArrayT
    ret["$TupleT"] = TupleT

    return ret


def get_types():
    result = {}
    result.update(get_primitive_types())

    return result
