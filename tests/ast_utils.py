from vyper.ast.nodes import VyperNode


def deepequals(node: VyperNode, other: VyperNode):
    # checks two nodes are recursively equal, ignoring metadata
    # like line info.
    if not isinstance(other, type(node)):
        return False

    if isinstance(node, list):
        if len(node) != len(other):
            return False
        return all(deepequals(a, b) for a, b in zip(node, other))

    if not isinstance(node, VyperNode):
        return node == other

    if getattr(node, "node_id", None) != getattr(other, "node_id", None):
        return False
    for field_name in (i for i in node.get_fields() if i not in VyperNode.__slots__):
        lhs = getattr(node, field_name, None)
        rhs = getattr(other, field_name, None)
        if not deepequals(lhs, rhs):
            return False
    return True


def deepcheck(n1: VyperNode, n2: VyperNode, path="") -> list[str]:
    """
    Finds differences between two nodes, ignoring metadata like line info

    Returns
    -------
    ret: list[str]
        List of differences in the format <path to node>: <difference>
    """
    # checks two nodes are recursively equal, ignoring metadata
    # like line info.

    def because(s: str) -> str:
        return f"{path}: {s}"

    def flatten(xss: list[list]) -> list:
        return [x for xs in xss for x in xs]

    if not isinstance(n2, type(n1)):
        return [because(f"Not the same type: {type(n1)} vs {type(n2)}")]

    if isinstance(n1, list):
        if len(n1) != len(n2):
            return [because(f"List of different lengths: {len(n1)} vs {len(n2)}")]
        return flatten(
            [deepcheck(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(n1, n2))]
        )

    if not isinstance(n1, VyperNode):
        if n1 == n2:
            return []
        else:
            return [because(f"Nodes are different")]

    n1_id = getattr(n1, "node_id", None)
    n2_id = getattr(n2, "node_id", None)
    if n1_id != n2_id:
        return [because(f"Different node ids: {n1_id} vs {n2_id}")]

    field_names = (i for i in n1.get_fields() if i not in VyperNode.__slots__)

    return flatten(
        [
            deepcheck(getattr(n1, field, None), getattr(n2, field, None), f"{path}.{field}")
            for field in field_names
        ]
    )
