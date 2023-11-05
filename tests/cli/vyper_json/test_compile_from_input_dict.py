#!/usr/bin/env python3

from copy import deepcopy
from pathlib import PurePath

import pytest

import vyper
from vyper.cli.vyper_json import (
    TRANSLATE_MAP,
    compile_from_input_dict,
    exc_handler_raises,
    exc_handler_to_dict,
)
from vyper.exceptions import InvalidType, JSONError, SyntaxException

FOO_CODE = """
import contracts.bar as Bar

@external
def foo(a: address) -> bool:
    return Bar(a).bar(1)

@external
def baz() -> uint256:
    return self.balance
"""

BAR_CODE = """
@external
def bar(a: uint256) -> bool:
    return True
"""

BAD_SYNTAX_CODE = """
def bar()>:
"""

BAD_COMPILER_CODE = """
@external
def oopsie(a: uint256) -> bool:
    return 42
"""

BAR_ABI = [
    {
        "name": "bar",
        "outputs": [{"type": "bool", "name": "out"}],
        "inputs": [{"type": "uint256", "name": "a"}],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

INPUT_JSON = {
    "language": "Vyper",
    "sources": {
        "contracts/foo.vy": {"content": FOO_CODE},
        "contracts/bar.vy": {"content": BAR_CODE},
    },
    "interfaces": {"contracts/ibar.json": {"abi": BAR_ABI}},
    "settings": {"outputSelection": {"*": ["*"]}},
}


def test_root_folder_not_exists():
    with pytest.raises(FileNotFoundError):
        compile_from_input_dict(INPUT_JSON, root_folder="/path/that/does/not/exist")


def test_wrong_language():
    with pytest.raises(JSONError):
        compile_from_input_dict({"language": "Solidity"})


def test_exc_handler_raises_syntax():
    input_json = deepcopy(INPUT_JSON)
    input_json["sources"]["badcode.vy"] = {"content": BAD_SYNTAX_CODE}
    with pytest.raises(SyntaxException):
        compile_from_input_dict(input_json, exc_handler_raises)


def test_exc_handler_to_dict_syntax():
    input_json = deepcopy(INPUT_JSON)
    input_json["sources"]["badcode.vy"] = {"content": BAD_SYNTAX_CODE}
    result, _ = compile_from_input_dict(input_json, exc_handler_to_dict)
    assert "errors" in result
    assert len(result["errors"]) == 1
    error = result["errors"][0]
    assert error["component"] == "compiler", error
    assert error["type"] == "SyntaxException"


def test_exc_handler_raises_compiler():
    input_json = deepcopy(INPUT_JSON)
    input_json["sources"]["badcode.vy"] = {"content": BAD_COMPILER_CODE}
    with pytest.raises(InvalidType):
        compile_from_input_dict(input_json, exc_handler_raises)


def test_exc_handler_to_dict_compiler():
    input_json = deepcopy(INPUT_JSON)
    input_json["sources"]["badcode.vy"] = {"content": BAD_COMPILER_CODE}
    result, _ = compile_from_input_dict(input_json, exc_handler_to_dict)
    assert sorted(result.keys()) == ["compiler", "errors"]
    assert result["compiler"] == f"vyper-{vyper.__version__}"
    assert len(result["errors"]) == 1
    error = result["errors"][0]
    assert error["component"] == "compiler"
    assert error["type"] == "InvalidType"


def test_source_ids_increment():
    input_json = deepcopy(INPUT_JSON)
    input_json["settings"]["outputSelection"] = {"*": ["evm.deployedBytecode.sourceMap"]}
    result, _ = compile_from_input_dict(input_json)
    assert result[PurePath("contracts/foo.vy")]["source_map"]["pc_pos_map_compressed"].startswith(
        "-1:-1:0"
    )
    assert result[PurePath("contracts/bar.vy")]["source_map"]["pc_pos_map_compressed"].startswith(
        "-1:-1:1"
    )


def test_outputs():
    result, _ = compile_from_input_dict(INPUT_JSON)
    assert list(result.keys()) == [PurePath("contracts/foo.vy"), PurePath("contracts/bar.vy")]

    output_formats = list(TRANSLATE_MAP.values())
    output_formats.append("source_id")
    assert sorted(result[PurePath("contracts/bar.vy")].keys()) == sorted(set(output_formats))


def test_relative_import_paths():
    input_json = deepcopy(INPUT_JSON)
    input_json["sources"]["contracts/potato/baz/baz.vy"] = {"content": """from ... import foo"""}
    input_json["sources"]["contracts/potato/baz/potato.vy"] = {"content": """from . import baz"""}
    input_json["sources"]["contracts/potato/footato.vy"] = {"content": """from baz import baz"""}
    compile_from_input_dict(input_json)
