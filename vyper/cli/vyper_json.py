#!/usr/bin/env python3

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Union

import vyper
from vyper.compiler.input_bundle import JSONInputBundle
from vyper.compiler.settings import OptimizationLevel, Settings
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.exceptions import JSONError
from vyper.typing import ContractCodes
from vyper.utils import keccak256

TRANSLATE_MAP = {
    "abi": "abi",
    "ast": "ast_dict",
    "devdoc": "devdoc",
    "evm.methodIdentifiers": "method_identifiers",
    "evm.bytecode.object": "bytecode",
    "evm.bytecode.opcodes": "opcodes",
    "evm.deployedBytecode.object": "bytecode_runtime",
    "evm.deployedBytecode.opcodes": "opcodes_runtime",
    "evm.deployedBytecode.sourceMap": "source_map",
    "evm.deployedBytecode.sourceMapFull": "source_map_full",
    "interface": "interface",
    "ir": "ir_dict",
    "ir_runtime": "ir_runtime_dict",
    "metadata": "metadata",
    "layout": "layout",
    "userdoc": "userdoc",
}


def _parse_cli_args():
    return _parse_args(sys.argv[1:])


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Vyper programming language for EVM - JSON Compiler",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="JSON file to compile from. If none is given, Vyper will receive it from stdin.",
        nargs="?",
    )
    parser.add_argument(
        "--version", action="version", version=f"{vyper.__version__}+commit.{vyper.__commit__}"
    )
    parser.add_argument(
        "-o",
        help="Filename to save JSON output to. If the file exists it will be overwritten.",
        default=None,
        dest="output_file",
    )
    parser.add_argument(
        "-p",
        help="Set a base import path. Vyper searches here if a file is not found in the JSON.",
        default=None,
        dest="root_folder",
    )
    parser.add_argument("--pretty-json", help="Output JSON in pretty format.", action="store_true")
    parser.add_argument(
        "--traceback",
        help="Show python traceback on error instead of returning JSON",
        action="store_true",
    )

    args = parser.parse_args(argv)
    if args.input_file:
        with Path(args.input_file).open() as fh:
            input_json = fh.read()
        json_path = Path(args.input_file).resolve().as_posix()
    else:
        input_json = "".join(sys.stdin.read()).strip()
        json_path = "<stdin>"

    exc_handler = exc_handler_raises if args.traceback else exc_handler_to_dict
    output_json = json.dumps(
        compile_json(input_json, exc_handler, args.root_folder, json_path),
        indent=2 if args.pretty_json else None,
        sort_keys=True,
        default=str,
    )

    if args.output_file is not None:
        output_path = Path(args.output_file).resolve()
        with output_path.open("w") as fh:
            fh.write(output_json)
        print(f"Results saved to {output_path}")
    else:
        print(output_json)


def exc_handler_raises(file_path: Union[str, None], exception: Exception, component: str) -> None:
    if file_path:
        print(f"Unhandled exception in '{file_path}':")
    exception._exc_handler = True  # type: ignore
    raise exception


def exc_handler_to_dict(file_path: Union[str, None], exception: Exception, component: str) -> Dict:
    err_dict: Dict = {
        "type": type(exception).__name__,
        "component": component,
        "severity": "error",
        "message": str(exception).strip('"'),
    }
    if hasattr(exception, "message"):
        err_dict.update(
            {"message": exception.message, "formattedMessage": str(exception)}  # type: ignore
        )
    if file_path is not None:
        err_dict["sourceLocation"] = {"file": file_path}
        if getattr(exception, "lineno", None) is not None:
            err_dict["sourceLocation"].update(
                {
                    "lineno": exception.lineno,  # type: ignore
                    "col_offset": getattr(exception, "col_offset", None),
                }
            )

    output_json = {"compiler": f"vyper-{vyper.__version__}", "errors": [err_dict]}
    return output_json


def get_evm_version(input_dict: Dict) -> Optional[str]:
    if "settings" not in input_dict:
        return None

    # TODO: move this validation somewhere it can be reused more easily
    evm_version = input_dict["settings"].get("evmVersion")
    if evm_version is None:
        return None

    if evm_version in (
        "homestead",
        "tangerineWhistle",
        "spuriousDragon",
        "byzantium",
        "constantinople",
    ):
        raise JSONError("Vyper does not support pre-istanbul EVM versions")
    if evm_version not in EVM_VERSIONS:
        raise JSONError(f"Unknown EVM version - '{evm_version}'")

    return evm_version


def get_compilation_targets(input_dict: dict) -> ContractCodes:
    if "compilation_targets" in input_dict:
        pass

    ret: ContractCodes = {}
    for path, value in input_dict["sources"].items():
        path = Path(path)
        if "urls" in value:
            raise JSONError(f"{path} - 'urls' is not a supported field, use 'content' instead")
        if "content" not in value:
            raise JSONError(f"{path} missing required field - 'content'")
        if "keccak256" in value:
            hash_ = value["keccak256"].lower()
            if hash_.startswith("0x"):
                hash_ = hash_[2:]
            if hash_ != keccak256(value["content"].encode("utf-8")).hex():
                raise JSONError(
                    f"Calculated keccak of '{path}' does not match keccak given in input JSON"
                )
        if path in ret:
            raise JSONError(f"Contract namespace collision: {key}")
        ret[path] = value["content"]
    return ret


def get_input_dict_files(input_dict: dict) -> dict[str, str]:
    ret: dict = {}

    for path, value in input_dict.get("interfaces", {}).items():
        if key in interface_sources:
            raise JSONError(f"Interface namespace collision: {key}")

        ret[path] = value

    return interface_sources


def get_input_dict_output_formats(input_dict: Dict, contract_sources: ContractCodes) -> Dict:
    output_formats = {}
    for path, outputs in input_dict["settings"]["outputSelection"].items():
        if isinstance(outputs, dict):
            # if outputs are given in solc json format, collapse them into a single list
            outputs = set(x for i in outputs.values() for x in i)
        else:
            outputs = set(outputs)

        for key in [i for i in ("evm", "evm.bytecode", "evm.deployedBytecode") if i in outputs]:
            outputs.remove(key)
            outputs.update([i for i in TRANSLATE_MAP if i.startswith(key)])
        if "*" in outputs:
            outputs = TRANSLATE_MAP.values()
        else:
            try:
                outputs = [TRANSLATE_MAP[i] for i in outputs]
            except KeyError as e:
                raise JSONError(f"Invalid outputSelection - {e}")

        outputs = sorted(set(outputs))

        if path == "*":
            output_keys = list(contract_sources.keys())
        else:
            output_keys = [_standardize_path(path)]
            if output_keys[0] not in contract_sources:
                raise JSONError(f"outputSelection references unknown contract '{output_keys[0]}'")

        for key in output_keys:
            output_formats[key] = outputs

    return output_formats


def compile_from_input_dict(
    input_dict: Dict,
    exc_handler: Callable = exc_handler_raises,
    root_folder: Union[str, None] = None,
) -> Tuple[Dict, Dict]:
    root_path = None
    if root_folder is not None:
        root_path = Path(root_folder).resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Invalid root path - '{root_path.as_posix()}' does not exist")

    if input_dict["language"] != "Vyper":
        raise JSONError(f"Invalid language '{input_dict['language']}' - Only Vyper is supported.")

    evm_version = get_evm_version(input_dict)

    optimize = input_dict["settings"].get("optimize")
    if isinstance(optimize, bool):
        # bool optimization level for backwards compatibility
        warnings.warn(
            "optimize: <bool> is deprecated! please use one of 'gas', 'codesize', 'none'."
        )
        optimize = OptimizationLevel.default() if optimize else OptimizationLevel.NONE
    elif isinstance(optimize, str):
        optimize = OptimizationLevel.from_string(optimize)
    else:
        assert optimize is None

    settings = Settings(evm_version=evm_version, optimize=optimize)

    no_bytecode_metadata = not input_dict["settings"].get("bytecodeMetadata", True)

    contract_sources: ContractCodes = get_compilation_targets(input_dict)
    output_formats = get_input_dict_output_formats(input_dict, contract_sources)

    compiler_data, warning_data = {}, {}
    warnings.simplefilter("always")
    input_bundle = JSONInputBundle(["."], contract_sources)
    for id_, contract_path in enumerate(sorted(contract_sources)):
        with warnings.catch_warnings(record=True) as caught_warnings:
            try:
                data = vyper.compile_codes(
                    {contract_path: contract_sources[contract_path]},
                    input_bundle,
                    output_formats[contract_path],
                    initial_id=id_,
                    settings=settings,
                    no_bytecode_metadata=no_bytecode_metadata,
                )
            except Exception as exc:
                return exc_handler(contract_path, exc, "compiler"), {}
            compiler_data[contract_path] = data[contract_path]
            if caught_warnings:
                warning_data[contract_path] = caught_warnings

    return compiler_data, warning_data


def format_to_output_dict(compiler_data: Dict) -> Dict:
    output_dict: Dict = {"compiler": f"vyper-{vyper.__version__}", "contracts": {}, "sources": {}}
    for id_, (path, data) in enumerate(compiler_data.items()):
        path = str(path)  # Path is not json serializable
        output_dict["sources"][path] = {"id": id_}
        if "ast_dict" in data:
            output_dict["sources"][path]["ast"] = data["ast_dict"]["ast"]

        name = Path(path).stem
        output_dict["contracts"][path] = {name: {}}
        output_contracts = output_dict["contracts"][path][name]

        if "ir_dict" in data:
            output_contracts["ir"] = data["ir_dict"]

        for key in ("abi", "devdoc", "interface", "metadata", "userdoc"):
            if key in data:
                output_contracts[key] = data[key]

        if "method_identifiers" in data:
            output_contracts["evm"] = {"methodIdentifiers": data["method_identifiers"]}

        evm_keys = ("bytecode", "opcodes")
        if any(i in data for i in evm_keys):
            evm = output_contracts.setdefault("evm", {}).setdefault("bytecode", {})
            if "bytecode" in data:
                evm["object"] = data["bytecode"]
            if "opcodes" in data:
                evm["opcodes"] = data["opcodes"]

        pc_maps_keys = ("source_map", "source_map_full")
        if any(i + "_runtime" in data for i in evm_keys) or any(i in data for i in pc_maps_keys):
            evm = output_contracts.setdefault("evm", {}).setdefault("deployedBytecode", {})
            if "bytecode_runtime" in data:
                evm["object"] = data["bytecode_runtime"]
            if "opcodes_runtime" in data:
                evm["opcodes"] = data["opcodes_runtime"]
            if "source_map" in data:
                evm["sourceMap"] = data["source_map"]["pc_pos_map_compressed"]
            if "source_map_full" in data:
                evm["sourceMapFull"] = data["source_map_full"]

    return output_dict


# https://stackoverflow.com/a/49518779
def _raise_on_duplicate_keys(ordered_pairs: List[Tuple[Hashable, Any]]) -> Dict:
    """
    Raise JSONError if a duplicate key exists in provided ordered list
    of pairs, otherwise return a dict.
    """
    dict_out = {}
    for key, val in ordered_pairs:
        if key in dict_out:
            raise JSONError(f"Duplicate key: {key}")
        else:
            dict_out[key] = val
    return dict_out


def compile_json(
    input_json: Union[Dict, str],
    exc_handler: Callable = exc_handler_raises,
    root_path: Union[str, None] = None,
    json_path: Union[str, None] = None,
) -> Dict:
    try:
        if isinstance(input_json, str):
            try:
                input_dict: Dict = json.loads(
                    input_json, object_pairs_hook=_raise_on_duplicate_keys
                )
            except json.decoder.JSONDecodeError as exc:
                new_exc = JSONError(str(exc), exc.lineno, exc.colno)
                return exc_handler(json_path, new_exc, "json")
        else:
            input_dict = input_json

        try:
            compiler_data, warn_data = compile_from_input_dict(input_dict, exc_handler, root_path)
            if "errors" in compiler_data:
                return compiler_data
        except KeyError as exc:
            new_exc = JSONError(f"Input JSON missing required field: {str(exc)}")
            return exc_handler(json_path, new_exc, "json")
        except (FileNotFoundError, JSONError) as exc:
            return exc_handler(json_path, exc, "json")

        output_dict = format_to_output_dict(compiler_data)
        if warn_data:
            output_dict["errors"] = []
            for path, msg in ((k, x) for k, v in warn_data.items() for x in v):
                output_dict["errors"].append(
                    {
                        "type": msg.category.__name__,
                        "component": "compiler",
                        "severity": "warning",
                        "message": msg.message,
                        "sourceLocation": {"file": path},
                    }
                )
        return output_dict

    except Exception as exc:
        if hasattr(exc, "_exc_handler"):
            # exception was already handled by exc_handler_raises
            raise
        exc.lineno = sys.exc_info()[-1].tb_lineno  # type: ignore
        file_path = sys.exc_info()[-1].tb_frame.f_code.co_filename  # type: ignore
        return exc_handler(file_path, exc, "vyper")
