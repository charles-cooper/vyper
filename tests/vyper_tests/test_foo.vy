def foo(x: uint256) -> uint256:
    return x * 5

--------------------------------

cases:
  foo:
    - input: 100
      output: 500
