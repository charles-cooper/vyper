

class VyperIR:
    pass

class VariableDecl(VyperIR):
    pass

class Copy(VyperIR):
    pass

class IRBuilder:
    def variable(self, name, location):
        return VariableDecl(name, location)

    def copy(self, dst, src):
        return Copy(dst, src)
