import json

import pytest
from eth_utils import to_wei

from tests.utils import decimal_to_int
from vyper.compiler import compile_code, compile_from_file_input
from vyper.exceptions import (
    ArgumentException,
    DuplicateImport,
    InterfaceViolation,
    NamespaceCollision,
)


# TODO CMC 2024-10-13: this should probably be in tests/unit/compiler/
def test_basic_extract_interface():
    code = """
# Events

event Transfer:
    _from: address
    _to: address
    _value: uint256


# Functions

@view
@external
def allowance(_owner: address, _spender: address) -> (uint256, uint256):
    return 1, 2
    """

    out = compile_code(code, output_formats=["interface"])
    out = out["interface"]
    code_pass = "\n".join(code.split("\n")[:-2] + ["    ..."])  # replace with a pass statement.

    assert code_pass.strip() == out.strip()


# TODO CMC 2024-10-13: this should probably be in tests/unit/compiler/
def test_basic_extract_external_interface():
    code = """
@view
@external
def allowance(_owner: address, _spender: address) -> (uint256, uint256):
    return 1, 2

@external
def test(_owner: address):
    pass

@view
@internal
def _prive(_owner: address, _spender: address) -> (uint256, uint256):
    return 1, 2
    """

    interface = """
# External Interfaces
interface One:
    def allowance(_owner: address, _spender: address) -> (uint256, uint256): view
    def test(_owner: address): nonpayable
    """

    out = compile_code(code, contract_path="One.vy", output_formats=["external_interface"])[
        "external_interface"
    ]

    assert interface.strip() == out.strip()


# TODO CMC 2024-10-13: should probably be in syntax tests
def test_basic_interface_implements(assert_compile_failed):
    code = """
from ethereum.ercs import IERC20

implements: IERC20

@external
def test() -> bool:
    return True
    """

    assert_compile_failed(lambda: compile_code(code), InterfaceViolation)


# TODO CMC 2024-10-13: should probably be in syntax tests
def test_external_interface_parsing(make_input_bundle, assert_compile_failed):
    interface_code = """
@external
def foo() -> uint256:
    ...

@external
def bar() -> uint256:
    ...
    """

    input_bundle = make_input_bundle({"a.vyi": interface_code})

    code = """
import a as FooBarInterface

implements: FooBarInterface

@external
def foo() -> uint256:
    return 1

@external
def bar() -> uint256:
    return 2
    """

    assert compile_code(code, input_bundle=input_bundle)

    not_implemented_code = """
import a as FooBarInterface

implements: FooBarInterface

@external
def foo() -> uint256:
    return 1

    """

    with pytest.raises(InterfaceViolation):
        compile_code(not_implemented_code, input_bundle=input_bundle)


# TODO CMC 2024-10-13: should probably be in syntax tests
def test_log_interface_event(make_input_bundle, assert_compile_failed):
    interface_code = """
event Foo:
    a: uint256
    """

    input_bundle = make_input_bundle({"a.vyi": interface_code})

    main = """
import a as FooBarInterface

implements: FooBarInterface

@external
def bar() -> uint256:
    log FooBarInterface.Foo(1)
    return 1
    """

    assert compile_code(main, input_bundle=input_bundle) is not None


VALID_IMPORT_CODE = [
    # import statement, import path without suffix
    ("import a as Foo", "a.vyi"),
    ("import b.a as Foo", "b/a.vyi"),
    ("import Foo as Foo", "Foo.vyi"),
    ("from a import Foo", "a/Foo.vyi"),
    ("from b.a import Foo", "b/a/Foo.vyi"),
]


# TODO CMC 2024-10-13: should probably be in syntax tests
@pytest.mark.parametrize("code,filename", VALID_IMPORT_CODE)
def test_extract_file_interface_imports(code, filename, make_input_bundle):
    input_bundle = make_input_bundle({filename: ""})

    assert compile_code(code, input_bundle=input_bundle) is not None


VALID_RELATIVE_IMPORT_CODE = [
    # import statement, import path without suffix
    ("from .a import Foo", "mock.vy"),
    ("from ..a import Foo", "b/mock.vy"),
]


# TODO CMC 2024-10-13: should probably be in syntax tests
@pytest.mark.parametrize("code,filename", VALID_RELATIVE_IMPORT_CODE)
def test_extract_file_interface_relative_imports(code, filename, make_input_bundle):
    input_bundle = make_input_bundle({"a/Foo.vyi": "", filename: code})

    file_input = input_bundle.load_file(filename)
    assert compile_from_file_input(file_input, input_bundle=input_bundle) is not None


BAD_IMPORT_CODE = [
    ("import a as A\nimport a as A", DuplicateImport),
    ("import a as A\nimport a as a", DuplicateImport),
    ("from . import a\nimport a as a", DuplicateImport),
    ("import a as a\nfrom . import a", DuplicateImport),
    ("from b import a\nfrom . import a", NamespaceCollision),
    ("import a\nimport c as a", NamespaceCollision),
]


# TODO CMC 2024-10-13: should probably be in syntax tests
@pytest.mark.parametrize("code,exception_type", BAD_IMPORT_CODE)
def test_extract_file_interface_imports_raises(code, exception_type, make_input_bundle):
    input_bundle = make_input_bundle({"a.vyi": "", "b/a.vyi": "", "c.vyi": "", "mock.vy": code})
    file_input = input_bundle.load_file("mock.vy")
    with pytest.raises(exception_type):
        compile_from_file_input(file_input, input_bundle=input_bundle)


def test_external_call_to_interface(env, get_contract, make_input_bundle):
    token_interface = """
@view
@external
def balanceOf(addr: address) -> uint256:
    ...

@external
def transfer(to: address, amount: uint256):
    ...
    """

    token_code = """
import itoken as IToken

implements: IToken

balanceOf: public(HashMap[address, uint256])

@external
def transfer(to: address, amount: uint256):
    self.balanceOf[to] += amount
    """

    input_bundle = make_input_bundle({"token.vy": token_code, "itoken.vyi": token_interface})

    code = """
import itoken as IToken

interface EPI:
    def test() -> uint256: view

token_address: IToken

@deploy
def __init__(_token_address: address):
    self.token_address = IToken(_token_address)

@external
def test():
    extcall self.token_address.transfer(msg.sender, 1000)
    """

    token = get_contract(token_code, input_bundle=input_bundle)

    test_c = get_contract(code, *[token.address], input_bundle=input_bundle)

    sender = env.deployer
    assert token.balanceOf(sender) == 0

    test_c.test()
    assert token.balanceOf(sender) == 1000


@pytest.mark.parametrize(
    "kwarg,typ,expected",
    [
        ("max_value(uint256)", "uint256", 2**256 - 1),
        ("min_value(int128)", "int128", -(2**127)),
        ("empty(uint8[2])", "uint8[2]", [0, 0]),
        ('method_id("vyper()", output_type=bytes4)', "bytes4", b"\x82\xcbE\xfb"),
        ("epsilon(decimal)", "decimal", decimal_to_int("1E-10")),
    ],
)
def test_external_call_to_interface_kwarg(get_contract, kwarg, typ, expected, make_input_bundle):
    interface_code = f"""
@external
@view
def foo(_max: {typ} = {kwarg}) -> {typ}:
    ...
    """
    code1 = f"""
import one as IContract

implements: IContract

@external
@view
def foo(_max: {typ} = {kwarg}) -> {typ}:
    return _max
    """

    input_bundle = make_input_bundle({"one.vyi": interface_code})

    code2 = f"""
import one as IContract

@external
@view
def bar(a_address: address) -> {typ}:
    return staticcall IContract(a_address).foo()
    """

    contract_a = get_contract(code1, input_bundle=input_bundle)
    contract_b = get_contract(code2, input_bundle=input_bundle)

    assert contract_b.bar(contract_a.address) == expected


def test_external_call_to_builtin_interface(env, get_contract):
    token_code = """
balanceOf: public(HashMap[address, uint256])

@external
def transfer(to: address, amount: uint256) -> bool:
    self.balanceOf[to] += amount
    return True
    """

    code = """
from ethereum.ercs import IERC20

token_address: IERC20

@deploy
def __init__(_token_address: address):
    self.token_address = IERC20(_token_address)

@external
def test():
    extcall self.token_address.transfer(msg.sender, 1000)
    """

    erc20 = get_contract(token_code)
    test_c = get_contract(code, *[erc20.address])

    sender = env.deployer
    assert erc20.balanceOf(sender) == 0

    test_c.test()
    assert erc20.balanceOf(sender) == 1000


def test_address_member(env, get_contract):
    code = """
interface Foo:
    def foo(): payable

f: Foo

@external
def test(addr: address):
    self.f = Foo(addr)
    assert self.f.address == addr
    """
    c = get_contract(code)
    for address in env.accounts:
        c.test(address)


# test data returned from external interface gets clamped
@pytest.mark.parametrize("typ", ("int128", "uint8"))
def test_external_interface_int_clampers(get_contract, tx_failed, typ):
    external_contract = f"""
@external
def ok() -> {typ}:
    return 1

@external
def should_fail() -> int256:
    return min_value(int256)
    """

    code = f"""
interface BadContract:
    def ok() -> {typ}: view
    def should_fail() -> {typ}: view

foo: BadContract

@deploy
def __init__(addr: BadContract):
    self.foo = addr


@external
def test_ok() -> {typ}:
    return staticcall self.foo.ok()

@external
def test_fail() -> {typ}:
    return staticcall self.foo.should_fail()

@external
def test_fail2() -> {typ}:
    x: {typ} = staticcall self.foo.should_fail()
    return x

@external
def test_fail3() -> int256:
    return convert(staticcall self.foo.should_fail(), int256)
    """

    bad_c = get_contract(external_contract)
    c = get_contract(code, bad_c.address)
    assert bad_c.ok() == 1
    assert bad_c.should_fail() == -(2**255)

    assert c.test_ok() == 1
    with tx_failed():
        c.test_fail()
    with tx_failed():
        c.test_fail2()
    with tx_failed():
        c.test_fail3()


# test data returned from external interface gets clamped
def test_external_interface_bytes_clampers(get_contract, tx_failed):
    external_contract = """
@external
def ok() -> Bytes[2]:
    return b"12"

@external
def should_fail() -> Bytes[3]:
    return b"123"
    """

    code = """
interface BadContract:
    def ok() -> Bytes[2]: view
    def should_fail() -> Bytes[2]: view

foo: BadContract

@deploy
def __init__(addr: BadContract):
    self.foo = addr


@external
def test_ok() -> Bytes[2]:
    return staticcall self.foo.ok()

@external
def test_fail1() -> Bytes[3]:
    return staticcall self.foo.should_fail()

@external
def test_fail2() -> Bytes[3]:
    return concat(staticcall self.foo.should_fail(), b"")
    """

    bad_c = get_contract(external_contract)
    c = get_contract(code, bad_c.address)
    assert bad_c.ok() == b"12"
    assert bad_c.should_fail() == b"123"

    assert c.test_ok() == b"12"
    with tx_failed():
        c.test_fail1()
    with tx_failed():
        c.test_fail2()


# test data returned from external interface gets clamped
def test_json_abi_bytes_clampers(get_contract, tx_failed, assert_compile_failed, make_input_bundle):
    external_contract = """
@external
def returns_Bytes3() -> Bytes[3]:
    return b"123"
    """

    should_not_compile = """
import BadJSONInterface
@external
def foo(x: BadJSONInterface) -> Bytes[2]:
    return slice(extcall x.returns_Bytes3(), 0, 2)
    """

    code = """
import BadJSONInterface

foo: BadJSONInterface

@deploy
def __init__(addr: BadJSONInterface):
    self.foo = addr

@external
def test_fail1() -> Bytes[2]:
    # should compile, but raise runtime exception
    return extcall self.foo.returns_Bytes3()

@external
def test_fail2() -> Bytes[2]:
    # should compile, but raise runtime exception
    x: Bytes[2] = extcall self.foo.returns_Bytes3()
    return x

@external
def test_fail3() -> Bytes[3]:
    # should revert - returns_Bytes3 is inferred to have return type Bytes[2]
    # (because test_fail3 comes after test_fail1)
    return extcall self.foo.returns_Bytes3()
    """

    bad_c = get_contract(external_contract)

    bad_json_interface = json.dumps(compile_code(external_contract, output_formats=["abi"])["abi"])
    input_bundle = make_input_bundle({"BadJSONInterface.json": bad_json_interface})

    assert_compile_failed(
        lambda: get_contract(should_not_compile, input_bundle=input_bundle), ArgumentException
    )

    c = get_contract(code, bad_c.address, input_bundle=input_bundle)
    assert bad_c.returns_Bytes3() == b"123"

    with tx_failed():
        c.test_fail1()
    with tx_failed():
        c.test_fail2()
    with tx_failed():
        c.test_fail3()


def test_units_interface(env, get_contract, make_input_bundle):
    code = """
import balanceof as BalanceOf

implements: BalanceOf

@external
@view
def balanceOf(owner: address) -> uint256:
    return as_wei_value(1, "ether")
    """

    interface_code = """
@external
@view
def balanceOf(owner: address) -> uint256:
    ...
    """

    input_bundle = make_input_bundle({"balanceof.vyi": interface_code})

    c = get_contract(code, input_bundle=input_bundle)

    assert c.balanceOf(env.deployer) == to_wei(1, "ether")


def test_simple_implements(make_input_bundle):
    interface_code = """
@external
def foo() -> uint256:
    ...
    """

    code = """
import a as FooInterface

implements: FooInterface

@external
def foo() -> uint256:
    return 1
    """

    input_bundle = make_input_bundle({"a.vyi": interface_code})

    assert compile_code(code, input_bundle=input_bundle) is not None


def test_self_interface_is_allowed(get_contract):
    code = """
interface Bar:
    def foo() -> uint256: view

@external
def foo() -> uint256 :
    return 42

@external
def bar() -> uint256:
    return staticcall Bar(self).foo()
"""
    c = get_contract(code)
    assert c.bar() == 42


def test_self_interface_via_storage(get_contract):
    code = """
interface Bar:
    def foo() -> uint256: view

bar_contract: Bar

@deploy
def __init__():
    self.bar_contract = Bar(self)

@external
def foo() -> uint256 :
    return 42

@external
def bar() -> uint256:
    return staticcall self.bar_contract.foo()
    """
    c = get_contract(code)
    assert c.bar() == 42


def test_self_interface_via_calldata(get_contract):
    code = """
interface Bar:
    def foo() -> uint256: view

@external
def foo() -> uint256 :
    return 42

@external
def bar(a: address) -> uint256:
    return staticcall Bar(a).foo()
    """
    c = get_contract(code)
    assert c.bar(c.address) == 42


type_str_params = [
    ("int128", -33),
    ("uint256", 42),
    ("bool", True),
    ("address", "0x1234567890123456789012345678901234567890"),
    ("bytes32", b"bytes32bytes32bytes32bytes32poop"),
    ("decimal", decimal_to_int("3.1337")),
    ("Bytes[4]", b"newp"),
    ("String[6]", "potato"),
]

interface_test_code = """
@external
@view
def test_json(a: {0}) -> {0}:
    return a
    """


def convert_v1_abi(abi):
    new_abi = []
    for func_abi in abi:
        if "stateMutability" in func_abi:
            mutability = func_abi["stateMutability"]
            del func_abi["stateMutability"]
            if mutability == "payable":
                func_abi["constant"] = False
                func_abi["payable"] = True
            elif mutability == "view":
                func_abi["constant"] = True
                func_abi["payable"] = False
            elif mutability == "pure":
                # NOTE: pure "changes" to "view"
                func_abi["constant"] = True
                func_abi["payable"] = False
            else:  # "nonpayable"
                func_abi["constant"] = False
                func_abi["payable"] = False
        else:  # Assume "nonpayable" by default
            func_abi["constant"] = False
            func_abi["payable"] = False
        new_abi.append(func_abi)
    return new_abi


@pytest.mark.parametrize("type_str", [i[0] for i in type_str_params])
def test_json_interface_implements(type_str, make_input_bundle, make_file):
    code = interface_test_code.format(type_str)

    abi = compile_code(code, output_formats=["abi"])["abi"]

    code = f"import jsonabi as jsonabi\nimplements: jsonabi\n{code}"

    input_bundle = make_input_bundle({"jsonabi.json": json.dumps(abi)})

    compile_code(code, input_bundle=input_bundle)

    # !!! overwrite the file
    make_file("jsonabi.json", json.dumps(convert_v1_abi(abi)))

    compile_code(code, input_bundle=input_bundle)


@pytest.mark.parametrize("type_str,value", type_str_params)
def test_json_interface_calls(get_contract, type_str, value, make_input_bundle, make_file):
    code = interface_test_code.format(type_str)

    abi = compile_code(code, output_formats=["abi"])["abi"]
    c1 = get_contract(code)

    code = f"""
import jsonabi as jsonabi

@external
@view
def test_call(a: address, b: {type_str}) -> {type_str}:
    return staticcall jsonabi(a).test_json(b)
    """
    input_bundle = make_input_bundle({"jsonabi.json": json.dumps(abi)})

    c2 = get_contract(code, input_bundle=input_bundle)
    assert c2.test_call(c1.address, value) == value

    make_file("jsonabi.json", json.dumps(convert_v1_abi(abi)))
    c3 = get_contract(code, input_bundle=input_bundle)
    assert c3.test_call(c1.address, value) == value


def test_interface_function_without_visibility(make_input_bundle, get_contract):
    interface_code = """
def foo() -> uint256:
    ...

@external
def bar() -> uint256:
    ...
    """

    code = """
import a as FooInterface

implements: FooInterface

@external
def foo() -> uint256:
    return 1

@external
def bar() -> uint256:
    return 1
    """

    input_bundle = make_input_bundle({"a.vyi": interface_code})

    c = get_contract(code, input_bundle=input_bundle)

    assert c.foo() == c.bar() == 1


def test_interface_with_structures():
    code = """
struct MyStruct:
    a: address
    b: uint256

event Transfer:
    sender: indexed(address)
    receiver: indexed(address)
    value: uint256

struct Voter:
    weight: int128
    voted: bool
    delegate: address
    vote: int128

@external
def bar():
    pass

event Buy:
    buyer: indexed(address)
    buy_order: uint256

@external
@view
def foo(s: MyStruct) -> MyStruct:
    return s
    """

    out = compile_code(code, contract_path="code.vy", output_formats=["interface"])["interface"]

    assert "# Structs" in out
    assert "struct MyStruct:" in out
    assert "b: uint256" in out
    assert "struct Voter:" in out
    assert "voted: bool" in out


def test_intrinsic_interface_instantiation(make_input_bundle, get_contract):
    lib1 = """
@external
@view
def foo():
    pass
    """
    main = """
import lib1

i: lib1.__interface__

@external
def bar() -> lib1.__interface__:
    self.i = lib1.__at__(self)
    return self.i
    """
    input_bundle = make_input_bundle({"lib1.vy": lib1})
    c = get_contract(main, input_bundle=input_bundle)

    assert c.bar() == c.address


def test_intrinsic_interface_converts(make_input_bundle, get_contract):
    lib1 = """
@external
@view
def foo():
    pass
    """
    main = """
import lib1

@external
def bar() -> lib1.__interface__:
    return lib1.__at__(self)
    """
    input_bundle = make_input_bundle({"lib1.vy": lib1})
    c = get_contract(main, input_bundle=input_bundle)

    assert c.bar() == c.address


def test_intrinsic_interface_kws(env, make_input_bundle, get_contract):
    value = 10**5
    lib1 = f"""
@external
@payable
def foo(a: address):
    send(a, {value})
    """
    main = f"""
import lib1

exports: lib1.__interface__

@external
def bar(a: address):
    extcall lib1.__at__(self).foo(a, value={value})
    """
    input_bundle = make_input_bundle({"lib1.vy": lib1})
    c = get_contract(main, input_bundle=input_bundle)
    env.set_balance(c.address, value)
    original_balance = env.get_balance(env.deployer)
    c.bar(env.deployer)
    assert env.get_balance(env.deployer) == original_balance + value


def test_intrinsic_interface_defaults(env, make_input_bundle, get_contract):
    lib1 = """
@external
@payable
def foo(i: uint256=1) -> uint256:
    return i
    """
    main = """
import lib1

exports: lib1.__interface__

@external
def bar() -> uint256:
    return extcall lib1.__at__(self).foo()
    """
    input_bundle = make_input_bundle({"lib1.vy": lib1})
    c = get_contract(main, input_bundle=input_bundle)
    assert c.bar() == 1


def test_interface_with_flags():
    code = """
struct MyStruct:
    a: address

flag Foo:
    BOO
    MOO
    POO

event Transfer:
    sender: indexed(address)

@external
def bar():
    pass
flag BAR:
    BIZ
    BAZ
    BOO

@external
@view
def foo(s: MyStruct) -> MyStruct:
    return s
    """

    out = compile_code(code, contract_path="code.vy", output_formats=["interface"])["interface"]

    assert "# Flags" in out
    assert "flag Foo:" in out
    assert "flag BAR" in out
    assert "BOO" in out
    assert "MOO" in out

    compile_code(out, contract_path="code.vyi", output_formats=["interface"])


vyi_filenames = [
    "test__test.vyi",
    "test__t.vyi",
    "t__test.vyi",
    "t__t.vyi",
    "t_t.vyi",
    "test_test.vyi",
    "t_test.vyi",
    "test_t.vyi",
    "_test_t__t_tt_.vyi",
    "foo_bar_baz.vyi",
]


@pytest.mark.parametrize("vyi_filename", vyi_filenames)
def test_external_interface_names(vyi_filename):
    code = """
@external
def foo():
    ...
    """

    compile_code(code, contract_path=vyi_filename, output_formats=["external_interface"])


def test_external_interface_with_flag():
    code = """
flag Foo:
    Blah

@external
def foo() -> Foo:
    ...
    """

    out = compile_code(code, contract_path="test__test.vyi", output_formats=["external_interface"])[
        "external_interface"
    ]
    assert "-> Foo:" in out


def test_external_interface_compiles_again():
    code = """
@external
def foo() -> uint256:
    ...
@external
def bar(a:int32) -> uint256:
    ...
    """

    out = compile_code(code, contract_path="test.vyi", output_formats=["external_interface"])[
        "external_interface"
    ]
    compile_code(out, contract_path="test.vyi", output_formats=["external_interface"])


@pytest.mark.xfail
def test_weird_interface_name():
    # based on comment https://github.com/vyperlang/vyper/pull/4290#discussion_r1884137428
    # we replace "_" for "" which results in an interface without name
    out = compile_code("", contract_path="_.vyi", output_formats=["external_interface"])[
        "external_interface"
    ]
    assert "interface _:" in out
