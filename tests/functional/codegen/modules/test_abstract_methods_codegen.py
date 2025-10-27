"""
Functional tests for abstract methods - verifying the generated code works correctly
"""
import pytest
from decimal import Decimal


def test_basic_abstract_method_execution(get_contract, make_input_bundle):
    """Check that abstract methods are properly inlined and executed"""
    base_module = """
@abstract
def get_multiplier() -> uint256: ...

@external
def calculate(x: uint256) -> uint256:
    return x * self.get_multiplier()
    """
    
    contract = """
import base

resolutions:
    base.get_multiplier -> self._get_mult

@internal
def _get_mult() -> uint256:
    return 10
    """
    
    c = get_contract(contract, input_bundle=make_input_bundle({"base.vy": base_module}))
    
    # should return x * 10
    assert c.calculate(5) == 50
    assert c.calculate(100) == 1000
    assert c.calculate(0) == 0


def test_abstract_method_with_state_access(get_contract, make_input_bundle):
    """Verify abstract methods can access contract state"""
    base_module = """
@abstract
def check_balance(account: address) -> bool: ...

@external
def can_withdraw(account: address, amount: uint256) -> bool:
    if not self.check_balance(account):
        return False
    return self.balances[account] >= amount

balances: public(HashMap[address, uint256])
    """
    
    contract = """
import base

resolutions:
    base.check_balance -> self._has_balance

balances: public(HashMap[address, uint256])
min_balance: constant(uint256) = 100

@internal
def _has_balance(account: address) -> bool:
    return self.balances[account] >= min_balance

@external
def set_balance(account: address, amount: uint256):
    self.balances[account] = amount
    """
    
    c = get_contract(contract, input_bundle=make_input_bundle({"base.vy": base_module}))
    
    # set up some balances
    user1 = "0x1234567890123456789012345678901234567890"
    user2 = "0x2345678901234567890123456789012345678901"
    
    c.set_balance(user1, 150)
    c.set_balance(user2, 50)
    
    # user1 has >= 100, so check_balance returns True
    assert c.can_withdraw(user1, 120) == True
    assert c.can_withdraw(user1, 200) == False  # not enough balance
    
    # user2 has < 100, so check_balance returns False  
    assert c.can_withdraw(user2, 30) == False


def test_module_provided_implementation(get_contract, make_input_bundle):
    """Test that module-provided implementations work correctly"""
    base_module = """
@abstract
def validate(value: uint256) -> bool: ...

@external
def process(value: uint256) -> uint256:
    if self.validate(value):
        return value * 2
    else:
        return 0
    """
    
    validator_module = """
import base

provides:
    base.validate -> self._check_range

MIN_VALUE: constant(uint256) = 10
MAX_VALUE: constant(uint256) = 1000

@internal
def _check_range(value: uint256) -> bool:
    return value >= MIN_VALUE and value <= MAX_VALUE
    """
    
    contract = """
import validator

resolutions:
    base.validate: accept validator._check_range
    """
    
    c = get_contract(
        contract, 
        input_bundle=make_input_bundle({
            "base.vy": base_module,
            "validator.vy": validator_module
        })
    )
    
    # values in range [10, 1000] should be doubled
    assert c.process(5) == 0      # too small
    assert c.process(10) == 20    # min value
    assert c.process(500) == 1000 # in range
    assert c.process(1000) == 2000 # max value
    assert c.process(1001) == 0   # too large


def test_override_provided_implementation(get_contract, make_input_bundle):
    """Make sure we can override a provided implementation"""
    base_module = """
@abstract
def get_fee() -> uint256: ...

@external
def calculate_total(amount: uint256) -> uint256:
    return amount + self.get_fee()
    """
    
    default_fee_module = """
import base

provides:
    base.get_fee -> self._default_fee

DEFAULT_FEE: constant(uint256) = 100

@internal
def _default_fee() -> uint256:
    return DEFAULT_FEE
    """
    
    # override with custom implementation
    contract = """
import default_fee

resolutions:
    base.get_fee: override default_fee._default_fee -> self._custom_fee

fee_rate: uint256

@deploy
def __init__():
    self.fee_rate = 50

@internal
def _custom_fee() -> uint256:
    return self.fee_rate

@external
def set_fee(new_fee: uint256):
    self.fee_rate = new_fee
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({
            "base.vy": base_module,
            "default_fee.vy": default_fee_module
        })
    )
    
    # using our custom fee of 50 (not the default 100)
    assert c.calculate_total(1000) == 1050
    
    # change fee
    c.set_fee(75)
    assert c.calculate_total(1000) == 1075


def test_diamond_resolution_choosing_impl(get_contract, make_input_bundle):
    """Test choosing between multiple implementations in diamond scenario"""
    base_module = """
@abstract  
def compute() -> uint256: ...

@external
def get_result() -> uint256:
    return self.compute()
    """
    
    impl_add_module = """
import base

provides:
    base.compute -> self._add_impl

value: uint256

@deploy
def __init__():
    self.value = 100

@internal
def _add_impl() -> uint256:
    return self.value + 50
    """
    
    impl_mul_module = """
import base

provides:
    base.compute -> self._mul_impl

value: uint256

@deploy
def __init__():
    self.value = 100

@internal
def _mul_impl() -> uint256:
    return self.value * 2
    """
    
    # contract choosing the add implementation
    contract_add = """
import impl_add
import impl_mul

resolutions:
    base.compute: accept impl_add._add_impl

value: uint256

@deploy
def __init__():
    self.value = 100
    """
    
    # contract choosing the mul implementation  
    contract_mul = """
import impl_add
import impl_mul

resolutions:
    base.compute: accept impl_mul._mul_impl

value: uint256

@deploy
def __init__():
    self.value = 100
    """
    
    input_bundle = make_input_bundle({
        "base.vy": base_module,
        "impl_add.vy": impl_add_module,
        "impl_mul.vy": impl_mul_module
    })
    
    c_add = get_contract(contract_add, input_bundle=input_bundle)
    c_mul = get_contract(contract_mul, input_bundle=input_bundle)
    
    assert c_add.get_result() == 150  # 100 + 50
    assert c_mul.get_result() == 200  # 100 * 2


def test_multiple_abstract_methods_integration(get_contract, make_input_bundle):
    """Test contract with multiple abstract methods from different modules"""
    auth_module = """
@abstract
def is_authorized(user: address) -> bool: ...

@external
def protected_action(user: address) -> bool:
    return self.is_authorized(user)
    """
    
    limits_module = """
@abstract
def check_limit(amount: uint256) -> bool: ...

@external  
def validate_amount(amount: uint256) -> bool:
    return self.check_limit(amount)
    """
    
    contract = """
import auth
import limits

resolutions:
    auth.is_authorized -> self._check_auth
    limits.check_limit -> self._check_lim

authorized: HashMap[address, bool]
max_amount: uint256

@deploy
def __init__():
    self.max_amount = 10000
    # authorize deployer
    self.authorized[msg.sender] = True

@internal
def _check_auth(user: address) -> bool:
    return self.authorized[user]

@internal
def _check_lim(amount: uint256) -> bool:
    return amount <= self.max_amount

@external
def authorize(user: address):
    # only deployer can authorize others (simplified)
    assert self.authorized[msg.sender], "not authorized"
    self.authorized[user] = True

@external
def do_transfer(recipient: address, amount: uint256) -> bool:
    # combines both abstract methods
    if not self.protected_action(msg.sender):
        return False
    if not self.validate_amount(amount):
        return False
    # ... actual transfer logic would go here
    return True
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({
            "auth.vy": auth_module,
            "limits.vy": limits_module
        })
    )
    
    user1 = "0x1234567890123456789012345678901234567890"
    
    # deployer is authorized, can transfer within limit
    assert c.do_transfer(user1, 5000) == True
    
    # deployer can't transfer over limit
    assert c.do_transfer(user1, 15000) == False
    
    # unauthorized user can't transfer even small amount
    # (need to send from different address - simplified test assumes msg.sender check)


def test_nested_abstract_calls(get_contract, make_input_bundle):
    """abstract methods calling other abstract methods"""
    base_module = """
@abstract
def step1() -> uint256: ...

@abstract
def step2(x: uint256) -> uint256: ...

@abstract
def step3(x: uint256) -> uint256: ...

@external
def full_calculation() -> uint256:
    a: uint256 = self.step1()
    b: uint256 = self.step2(a)  
    c: uint256 = self.step3(b)
    return c
    """
    
    partial_module = """
import base

provides:
    base.step1 -> self._s1
    base.step2 -> self._s2

@internal
def _s1() -> uint256:
    return 5

@internal
def _s2(x: uint256) -> uint256:
    return x * 3  # 5 * 3 = 15
    """
    
    contract = """
import partial

resolutions:
    base.step1: accept partial._s1
    base.step2: accept partial._s2  
    base.step3 -> self._my_step3

@internal
def _my_step3(x: uint256) -> uint256:
    return x + 10  # 15 + 10 = 25
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({
            "base.vy": base_module,
            "partial.vy": partial_module
        })
    )
    
    assert c.full_calculation() == 25


def test_abstract_external_methods(get_contract, make_input_bundle):
    """abstract methods can be external and callable from outside"""
    callback_module = """
@abstract
@external
def on_receive(sender: address, amount: uint256): ...

@external
def trigger_callback(amount: uint256):
    # calls the abstract external method
    self.on_receive(msg.sender, amount)
    """
    
    contract = """
import callback

resolutions:
    callback.on_receive -> self.handle_receive

received_from: public(address)
received_amount: public(uint256)
receive_count: public(uint256)

@external
def handle_receive(sender: address, amount: uint256):
    self.received_from = sender
    self.received_amount = amount
    self.receive_count += 1
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({"callback.vy": callback_module})
    )
    
    # calling through trigger_callback
    c.trigger_callback(1234)
    assert c.receive_count() == 1
    assert c.received_amount() == 1234
    
    # should also be able to call handle_receive directly since it's external
    c.handle_receive("0x1234567890123456789012345678901234567890", 5678)
    assert c.receive_count() == 2
    assert c.received_amount() == 5678


def test_abstract_view_methods(get_contract, make_input_bundle):
    """abstract view methods should work without state changes"""
    calculator_module = """
@abstract
@view
def get_base_value() -> uint256: ...

@external
@view
def calculate_percentage(percent: uint256) -> uint256:
    # calculate percent of base value
    base: uint256 = self.get_base_value()
    return (base * percent) // 100
    """
    
    contract = """
import calculator

resolutions:
    calculator.get_base_value -> self._get_base

stored_value: uint256

@deploy
def __init__():
    self.stored_value = 1000

@internal
@view
def _get_base() -> uint256:
    return self.stored_value

@external
def update_value(new_value: uint256):
    self.stored_value = new_value
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({"calculator.vy": calculator_module})
    )
    
    # 50% of 1000 = 500
    assert c.calculate_percentage(50) == 500
    
    # 10% of 1000 = 100  
    assert c.calculate_percentage(10) == 100
    
    # update base value
    c.update_value(2000)
    
    # 50% of 2000 = 1000
    assert c.calculate_percentage(50) == 1000


def test_abstract_with_events(get_contract, make_input_bundle, get_logs):
    """abstract methods that emit events"""
    logger_module = """
event ActionLogged:
    actor: indexed(address)
    action: String[32]

@abstract
def log_action(action: String[32]): ...

@external
def do_something():
    self.log_action("something")
    """
    
    contract = """
import logger

resolutions:
    logger.log_action -> self._log

event ActionLogged:
    actor: indexed(address)
    action: String[32]

@internal
def _log(action: String[32]):
    log ActionLogged(msg.sender, action)
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({"logger.vy": logger_module})
    )
    
    tx = c.do_something()
    logs = get_logs(tx, c)
    
    assert len(logs) == 1
    assert logs[0].args.action == "something"


def test_complex_inheritance_chain(get_contract, make_input_bundle):
    """Test that complex chains of module dependencies work"""
    level1_module = """
@abstract
def base_op() -> uint256: ...

@external
def level1_calc() -> uint256:
    return self.base_op() + 1
    """
    
    level2_module = """
import level1

provides:
    level1.base_op -> self._l2_op

counter: uint256

@internal
def _l2_op() -> uint256:
    self.counter += 1
    return self.counter * 10

@external
def level2_calc() -> uint256:
    return level1.level1_calc() + 100
    """
    
    level3_module = """
import level2

# doesn't provide anything, just uses level2

@external
def level3_calc() -> uint256:
    return level2.level2_calc() * 2
    """
    
    contract = """
import level3

resolutions:
    level1.base_op: accept level2._l2_op

counter: uint256
    """
    
    c = get_contract(
        contract,
        input_bundle=make_input_bundle({
            "level1.vy": level1_module,
            "level2.vy": level2_module,
            "level3.vy": level3_module
        })
    )
    
    # first call: counter=1, base_op returns 10
    # level1_calc = 10 + 1 = 11
    # level2_calc = 11 + 100 = 111
    # level3_calc = 111 * 2 = 222
    assert c.level3_calc() == 222
    
    # second call: counter=2, base_op returns 20
    # level1_calc = 20 + 1 = 21
    # level2_calc = 21 + 100 = 121
    # level3_calc = 121 * 2 = 242
    assert c.level3_calc() == 242