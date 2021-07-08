# 🐍 `vyper.ir` 🐍

## Purpose

The `vyper.ir` module defines an Intermediate Representation (IR) for
Vyper which is lower level than the AST after it has passed all semantic
annotation and checking but higher level than LLL or EVM opcodes. It
defines building blocks for compiler writers to emit output bytecode.

Its main goals are:
- Conceptually easy to translate from Vyper to VyperIR
- Simplify the Vyper compiler
- First step towards making it possible to target different backends easily
- Easy to transform to true SSA
- Enable certain classes of optimizations that are infeasible right now including
    - Reachability analysis
    - Dead code elimination
    - Constant folding
    - Bytecode size
    - Consolidating operations which operate on contiguous memory
    - Inlining
    - Caching


## Organization

stub

## Control Flow

stub

### Node Generation

stub

## Design

stub

### Interface Files (`.pyi`)

stub

This module makes use of Python interface files ("stubs") to aid in MyPy type
annotation.

Stubs share the same name as their source counterparts, with a `.pyi` extension.
Whenever included, a stub takes precedence over a source file. For example, given
the following file structure:

```bash
ast/
  node.py
  node.pyi
```

The type information in `node.pyi` is applied to `node.py`. Any types given in
`node.py` are only included to aid readability - they are ignored by the type
checker.

You must modify both the source file and the stub when you make changes to a source
file with a corresponding stub.

The following resources are useful for familiarizing yourself with stubs:

* [MyPy: Stub Files](https://mypy.readthedocs.io/en/stable/stubs.html)
* [PEP 484: Stub Files](https://www.python.org/dev/peps/pep-0484/#stub-files)

## Integration

stub
