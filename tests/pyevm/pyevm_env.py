import json
from typing import Callable

import eth.tools.builder.chain as chain
from eth_tester.exceptions import TransactionFailed
from eth import constants
from eth._utils.address import generate_contract_address
from eth.chains.mainnet import MainnetChain
from eth.db.atomic import AtomicDB
from eth.vm.message import Message
from eth.vm.transaction_context import BaseTransactionContext
from eth_keys.datatypes import PrivateKey
from eth_tester.backends.pyevm.main import get_default_account_keys
from eth_typing import HexAddress
from eth_utils import to_canonical_address, to_checksum_address
from hexbytes import HexBytes

from tests.pyevm.abi import abi_encode
from tests.pyevm.abi_contract import ABIContract, ABIContractFactory, ABIFunction
from vyper.ast.grammar import parse_vyper_source
from vyper.compiler import CompilerData, Settings, compile_code
from vyper.compiler.settings import OptimizationLevel
from vyper.utils import ERC5202_PREFIX


# TODO make fork configurable - ex. "latest", "frontier", "berlin"
# TODO make genesis params+state configurable
def _make_chain(gas_limit):
    if gas_limit is None:
        gas_limit = 1e18
    gas_limit = int(gas_limit)
    params = {"difficulty": constants.GENESIS_DIFFICULTY, "gas_limit": gas_limit}
    # TODO should we use MiningChain? is there a perf difference?
    # TODO debug why `fork_at()` cannot accept 0 as block num
    _Chain = chain.build(MainnetChain, chain.latest_mainnet_at(1))
    return _Chain.from_genesis(AtomicDB(), params)


class PyEVMEnv:
    def __init__(self, gas_limit: int | float | None, tracing=False) -> None:
        self.evm = _make_chain(gas_limit).get_vm()
        self.bytecode: dict[HexAddress, str] = {}
        self.contracts: dict[HexAddress, ABIContract] = {}
        self._keys: list[PrivateKey] = get_default_account_keys()

    def get_gas_price(self):
        return 0

    @property
    def accounts(self) -> list[HexAddress]:
        return [k.public_key.to_checksum_address() for k in self._keys]

    @property
    def deployer(self) -> HexAddress:
        return self._keys[0].public_key.to_checksum_address()

    def set_balance(self, address: HexAddress, value: int):
        self.evm.set_balance(address, value)

    def execute_code(
        self,
        to_address: HexAddress,
        sender: HexAddress,
        data: bytes | None,
        value: int | None,
        # todo: use or remove arguments
        gas: int,
        is_modifying: bool,
        contract: "ABIContract",
        transact=None,
    ):
        if gas is None:
            gas = self.evm.state.gas_limit

        sender = to_canonical_address(sender)
        to = to_canonical_address(to_address)

        bytecode = self.evm.state.get_code(to)
        is_static = not is_modifying
        msg = Message(
            sender=sender,
            to=to,
            gas=gas,
            value=value,
            code=bytecode,
            data=data,
            is_static=is_static,
        )
        origin = sender  # XXX: consider making this parametrizable
        tx_ctx = BaseTransactionContext(origin=origin, gas_price=self.get_gas_price())

        c = self.evm.state.computation_class.apply_message(self.evm.state, msg, tx_ctx)
        if c.is_error:
            raise TransactionFailed(c.error)
        return c

    def get_code(self, address: HexAddress):
        return self.evm.state.get_code(to_canonical_address(address))

    def register_contract(self, address: HexAddress, contract: "ABIContract"):
        self.contracts[address] = contract

    def deploy_source(
        self,
        source_code: str,
        optimize: OptimizationLevel,
        output_formats: dict[str, Callable[[CompilerData], str]],
        *args,
        override_opt_level=None,
        input_bundle=None,
        evm_version=None,
        **kwargs,
    ) -> ABIContract:
        abi, bytecode = self._compile(
            source_code, optimize, output_formats, override_opt_level, input_bundle, evm_version
        )
        value = (
            kwargs.pop("value", 0) or kwargs.pop("value_in_eth", 0) * 10**18
        )  # Handle deploying with an eth value.

        return self.deploy(abi, bytecode, value, *args, **kwargs)

    def _compile(
        self, source_code, optimize, output_formats, override_opt_level, input_bundle, evm_version
    ):
        out = compile_code(
            source_code,
            # test that all output formats can get generated
            output_formats=output_formats,
            settings=Settings(evm_version=evm_version, optimize=override_opt_level or optimize),
            input_bundle=input_bundle,
            show_gas_estimates=True,  # Enable gas estimates for testing
        )
        parse_vyper_source(source_code)  # Test grammar.
        json.dumps(out["metadata"])  # test metadata is json serializable
        return out["abi"], out["bytecode"]

    def deploy_blueprint(
        self,
        source_code,
        optimize,
        output_formats,
        *args,
        override_opt_level=None,
        input_bundle=None,
        evm_version=None,
        initcode_prefix=ERC5202_PREFIX,
    ):
        abi, bytecode = self._compile(
            source_code, optimize, output_formats, override_opt_level, input_bundle, evm_version
        )
        bytecode = HexBytes(initcode_prefix) + HexBytes(bytecode)
        bytecode_len = len(bytecode)
        bytecode_len_hex = hex(bytecode_len)[2:].rjust(4, "0")
        # prepend a quick deploy preamble
        deploy_preamble = HexBytes("61" + bytecode_len_hex + "3d81600a3d39f3")
        deploy_bytecode = HexBytes(deploy_preamble) + bytecode

        deployer_abi = []  # just a constructor
        deployer = self.deploy(deployer_abi, deploy_bytecode.hex())

        def factory(address):
            return ABIContractFactory.from_abi_dict(abi).at(self, address)

        return deployer, factory

    def deploy(self, abi: list[dict], bytecode: str, value=0, gas=None, *args, **kwargs):
        if gas is None:
            gas = self.evm.state.gas_limit

        factory = ABIContractFactory.from_abi_dict(abi=abi)

        initcode = bytes.fromhex(bytecode.removeprefix("0x"))

        if args or kwargs:
            ctor_abi = next(i for i in abi if i["type"] == "constructor")
            ctor = ABIFunction(ctor_abi, contract_name=factory._name)
            initcode += abi_encode(ctor.signature, ctor._merge_kwargs(*args, **kwargs))

        sender = to_canonical_address(self.deployer)
        nonce = self.evm.state.get_nonce(sender)
        self.evm.state.increment_nonce(sender)
        target_address = generate_contract_address(sender, nonce)

        msg = Message(
            to=constants.CREATE_CONTRACT_ADDRESS,  # i.e., b""
            sender=sender,
            gas=gas,
            value=value,
            code=initcode,
            create_address=target_address,
            data=b"",
        )

        origin = sender  # XXX: consider making this parametrizable
        tx_ctx = BaseTransactionContext(origin=origin, gas_price=self.get_gas_price())
        c = self.evm.state.computation_class.apply_create_message(self.evm.state, msg, tx_ctx)

        if c.is_error:
            raise TransactionFailed(c.error)

        address = to_checksum_address(target_address)
        abi_contract = factory.at(self, address)

        return abi_contract
