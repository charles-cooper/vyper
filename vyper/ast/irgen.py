# generate Vyper IR from a Vyper AST

import vyper.ast.nodes as vy_ast
from vyper.ast.visitor import VyperNodeVisitorBase
from vyper.ir.nodes import VyperIR


class IRGen(VyperNodeVisitorBase):
    def visit_Return(self, node):
        pass

    def visit_Log(self, node):
        pass

    def visit_EventDef(self, node):
        pass

    def visit_InterfaceDef(self, node):
        self._interfaces[node.name] = node

    def visit_StructDef(self, node):
        self._structs[node.name] = node

    def visit_Assign(self, node):
        dst = IRGen(node.target).ir()
        src = IRGen(node.value).ir()
        self.ir.copy(dst, src)

    def visit_AugAssign(self, node):
        dst = IRGen(node.target).ir()
        src = IRGen(node.value).ir()
        self.ir.copy(dst, _dispatch_binop(src.node.op, dst, src))

    def visit_For(self, node):
        x = self.ir.variable(target)
        self.ir.loop(COND, BODY)

    def visit_If(self, node):
        cond = self.ir.variable(node.test)
        self.ir.branch(cond, body, orelse)

    def visit_Call(self, node):
