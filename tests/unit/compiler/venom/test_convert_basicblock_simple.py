from vyper.codegen.ir_node import IRnode
from vyper.compiler.settings import OptimizationLevel
from vyper.venom import generate_ir


def test_simple():
    ir = ["calldatacopy", 32, 0, ["calldatasize"]]
    ir_node = IRnode.from_list(ir)
    deploy, runtime = generate_ir(ir_node, OptimizationLevel.NONE)
    assert deploy is None
    assert runtime is not None

    bb = runtime.basic_blocks[0]
    assert bb.instructions[0].opcode == "calldatasize"
    assert bb.instructions[1].opcode == "calldatacopy"


def test_simple_2():
    ir = [
        "seq",
        [
            "seq",
            [
                "mstore",
                ["add", 64, 0],
                [
                    "with",
                    "x",
                    ["calldataload", ["add", 4, 0]],
                    [
                        "with",
                        "ans",
                        ["add", "x", 1],
                        ["seq", ["assert", ["ge", "ans", "x"]], "ans"],
                    ],
                ],
            ],
        ],
        32,
    ]
    ir_node = IRnode.from_list(ir)
    deploy, runtime = generate_ir(ir_node, OptimizationLevel.NONE)
    assert deploy is None
    assert runtime is not None

    print(runtime)


if __name__ == "__main__":
    test_simple()
    test_simple_2()
