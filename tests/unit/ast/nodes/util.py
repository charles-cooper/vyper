from vyper.ast import VyperNode


# useful for testing
def deepequals(node: VyperNode, other: VyperNode):
    if not isinstance(other, type(node)):
        return False

    if getattr(node, "node_id", None) != getattr(other, "node_id", None):
        return False

    for field_name in node.get_fields():
        if field_name in VyperNode.__slots__:
            continue
        lhs = getattr(node, field_name, None)
        rhs = getattr(other, field_name, None)

        if isinstance(lhs, list) and isinstance(rhs, list):
            if len(lhs) != len(rhs):
                return False
            if not all(deepequals(x, y) for (x, y) in zip(lhs, rhs)):
                return False
        elif not isinstance(lhs, VyperNode):
            if lhs != rhs:
                return False
        elif not deepequals(lhs, rhs):
            return False

    return True
