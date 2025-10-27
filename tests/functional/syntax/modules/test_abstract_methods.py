"""
Tests for abstract methods feature based on PLAN.md
"""
import pytest
from vyper import compile_code
from vyper.exceptions import (
    StructureException,
    InvalidOperation,
    NamespaceCollision,
    InterfaceViolation,
    # These exception types will need to be added
    # UnresolvedAbstractMethod,
    # ConflictingProviders,
    # InvalidResolution,
)


def test_basic_abstract_method(make_input_bundle):
    """Test basic abstract method declaration and resolution"""
    base_module = """
@abstract
def validate_transfer(sender: address, recipient: address, amount: uint256): ...

@external
def transfer(recipient: address, amount: uint256) -> bool:
    self.validate_transfer(msg.sender, recipient, amount)
    self.balances[msg.sender] -= amount
    self.balances[recipient] += amount
    return True

balances: public(HashMap[address, uint256])
    """
    
    contract = """
import base

resolutions: (
        base.validate_transfer => self._validate,
)

@internal
def _validate(sender: address, recipient: address, amount: uint256):
    assert self.balances[sender] >= amount, "Insufficient balance"

balances: public(HashMap[address, uint256])
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module})
    compile_code(contract, input_bundle=input_bundle)


def test_module_provides_implementation(make_input_bundle):
    """Test module providing implementation for abstract method"""
    base_module = """
@abstract
def before_transfer(sender: address, recipient: address, amount: uint256): ...

@external
def transfer(recipient: address, amount: uint256) -> bool:
    self.before_transfer(msg.sender, recipient, amount)
    return True
    """
    
    pausable_module = """
import base

provides: (
        base.before_transfer => self._check_pause,
    )

paused: public(bool)

@internal
def _check_pause(sender: address, recipient: address, amount: uint256):
    assert not self.paused, "Token is paused"
    """
    
    contract = """
import base
import pausable

resolutions: (
            base.before_transfer => accept pausable._check_pause,
        )
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module, "pausable.vy": pausable_module})
    compile_code(contract, input_bundle=input_bundle)


def test_override_provided_implementation(make_input_bundle):
    """Test overriding a provided implementation"""
    base_module = """
@abstract
def validate_amount(amount: uint256): ...

@external
def transfer(amount: uint256) -> bool:
    self.validate_amount(amount)
    return True
    """
    
    min_check_module = """
import base

provides: (
        base.validate_amount => self._check_minimum,
    )

min_amount: uint256

@internal
def _check_minimum(amount: uint256):
    assert amount >= self.min_amount
    """
    
    contract = """
import base
import min_check

resolutions: (
            base.validate_amount => override self._my_validation[min_check._check_minimum],
        )

max_amount: uint256

@internal
def _my_validation(amount: uint256):
    assert amount > 0
    assert amount <= self.max_amount
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module, "min_check.vy": min_check_module})
    compile_code(contract, input_bundle=input_bundle)


def test_diamond_problem_explicit_resolution(make_input_bundle):
    """Test diamond problem with explicit resolution"""
    base_module = """
@abstract
def process(): ...

@external
def execute() -> bool:
    self.process()
    return True
    """
    
    impl_a_module = """
import base

provides: (
        base.process => self._process_a,
    )

@internal
def _process_a():
    pass
    """
    
    impl_b_module = """
import base

provides: (
        base.process => self._process_b,
    )

@internal
def _process_b():
    pass
    """
    
    # Using impl_a's implementation
    contract_a = """
import base
import impl_a
import impl_b

resolutions: (
            base.process => accept impl_a._process_a,
        )
    """
    
    # Using impl_b's implementation
    contract_b = """
import base
import impl_a
import impl_b

resolutions: (
            base.process => accept impl_b._process_b,
        )
    """
    
    # Overriding both
    contract_override = """
import base
import impl_a
import impl_b

resolutions: (
            base.process => override self._my_process[impl_a._process_a, impl_b._process_b],
        )

@internal
def _my_process():
    pass
    """
    
    # All should compile successfully
    input_bundle = make_input_bundle({
        "base.vy": base_module,
        "impl_a.vy": impl_a_module,
        "impl_b.vy": impl_b_module,
    })
    compile_code(contract_a, input_bundle=input_bundle)
    compile_code(contract_b, input_bundle=input_bundle)
    compile_code(contract_override, input_bundle=input_bundle)


def test_diamond_problem_missing_resolution(make_input_bundle):
    """Test diamond problem without resolution should fail"""
    base_module = """
@abstract
def process(): ...
    """
    
    impl_a_module = """
import base

provides: (
        base.process => self._process_a,
    )

@internal
def _process_a():
    pass
    """
    
    impl_b_module = """
import base

provides: (
        base.process => self._process_b,
    )

@internal
def _process_b():
    pass
    """
    
    # Missing explicit resolution
    contract = """
import impl_a
import impl_b

# No resolutions block!
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({
            "base.vy": base_module,
            "impl_a.vy": impl_a_module,
            "impl_b.vy": impl_b_module,
        })
        compile_code(contract, input_bundle=input_bundle)
    
    # Check for appropriate error message
    error_msg = str(exc_info.value)
    assert "Multiple providers for base.process" in error_msg or "Conflicting" in error_msg


def test_unresolved_abstract_method(make_input_bundle):
    """Test error when abstract method is not resolved"""
    base_module = """
@abstract
def validate(): ...

@abstract
def authorize(): ...

@external
def execute():
    self.validate()
    self.authorize()
    """
    
    contract = """
import base

resolutions: (
        base.validate => self._validate,
    )

@internal
def _validate():
    pass

# Missing resolution for base.authorize!
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module})
        compile_code(contract, input_bundle=input_bundle)
    
    assert "base.authorize" in str(exc_info.value) or "unresolved" in str(exc_info.value).lower()


def test_multiple_abstract_methods(make_input_bundle):
    """Test multiple abstract methods with different resolutions"""
    base_module = """
@abstract
def before_transfer(sender: address, recipient: address, amount: uint256): ...

@abstract
def after_transfer(sender: address, recipient: address, amount: uint256): ...

@abstract
def on_mint(recipient: address, amount: uint256): ...

@external
def transfer(recipient: address, amount: uint256) -> bool:
    self.before_transfer(msg.sender, recipient, amount)
    # transfer logic
    self.after_transfer(msg.sender, recipient, amount)
    return True

@external  
def mint(recipient: address, amount: uint256):
    self.on_mint(recipient, amount)
    """
    
    pausable_module = """
import base

provides: (
        base.before_transfer => self._check_pause,
    )

paused: bool

@internal
def _check_pause(sender: address, recipient: address, amount: uint256):
    assert not self.paused
    """
    
    logger_module = """
import base

provides: (
        base.after_transfer => self._log_transfer,
    )

@internal
def _log_transfer(sender: address, recipient: address, amount: uint256):
    pass
    """
    
    contract = """
import base
import pausable
import logger

resolutions: (
            base.before_transfer => accept pausable._check_pause,
            base.after_transfer  => accept logger._log_transfer,
            base.on_mint         => self._mint_checks,
        )

max_supply: uint256
total_supply: uint256

@internal
def _mint_checks(recipient: address, amount: uint256):
    assert self.total_supply + amount <= self.max_supply
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({
        "base.vy": base_module,
        "pausable.vy": pausable_module,
        "logger.vy": logger_module,
    })
    compile_code(contract, input_bundle=input_bundle)


def test_nested_module_dependencies(make_input_bundle):
    """Test transitive abstract method requirements"""
    base_module = """
@abstract
def validate(): ...

@internal
def execute():
    self.validate()
    """
    
    middle_module = """
import base

# Passes through base.validate requirement without providing it
@external
def middle_execute():
    base.execute()
    """
    
    contract = """
import base
import middle

resolutions: (
        base.validate => self._validate,
    )

@internal
def _validate():
    pass
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module, "middle.vy": middle_module})
    compile_code(contract, input_bundle=input_bundle)


def test_module_cannot_override_other_module(make_input_bundle):
    """Test that modules cannot override other modules' provisions"""
    base_module = """
@abstract
def process(): ...
    """
    
    impl_module = """
import base

provides: (
    base.process => self._impl,
)

@internal
def _impl():
    pass
    """
    
    # This module tries to override impl's provision - should fail
    override_module = """
import impl

provides: (
    base.process => self._different,  # ERROR: Cannot override impl's provision
)

@internal  
def _different():
    pass
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module, "impl.vy": impl_module})
        compile_code(override_module, input_bundle=input_bundle)
    
    error_msg = str(exc_info.value)
    assert "Cannot provide base.process" in error_msg or "Already provided" in error_msg


def test_invalid_accept_nonexistent_provider(make_input_bundle):
    """Test error when accepting non-existent provider"""
    base_module = """
@abstract
def validate(): ...
    """
    
    contract = """
import base

resolutions: (
        base.validate => accept nonexistent._method,  # ERROR: No such provider
    )
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module})
        compile_code(contract, input_bundle=input_bundle)
    
    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg or "No provider found" in error_msg


def test_invalid_override_wrong_provider_list(make_input_bundle):
    """Test error when override list doesn't match actual providers"""
    base_module = """
@abstract  
def process(): ...
    """
    
    impl_module = """
import base

provides: (
    base.process => self._impl,
)

@internal
def _impl():
    pass
    """
    
    contract = """
import impl

resolutions: (
        # Wrong provider in override list
        base.process => override self._my_impl[wrong._provider],
    )

@internal
def _my_impl():
    pass
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module, "impl.vy": impl_module})
        compile_code(contract, input_bundle=input_bundle)
    
    error_msg = str(exc_info.value)
    assert "Invalid override" in error_msg or "wrong._provider" in error_msg


def test_abstract_method_with_return_type(make_input_bundle):
    """Test abstract methods with return types"""
    base_module = """
@abstract
def calculate(x: uint256, y: uint256) -> uint256: ...

@external
def compute(a: uint256, b: uint256) -> uint256:
    return self.calculate(a, b) + 100
    """
    
    contract = """
import base

resolutions: (
        base.calculate => self._calc,
    )

@internal
def _calc(x: uint256, y: uint256) -> uint256:
    return x * y
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module})
    compile_code(contract, input_bundle=input_bundle)


def test_abstract_method_signature_mismatch(make_input_bundle):
    """Test error when implementation signature doesn't match abstract"""
    base_module = """
@abstract
def process(x: uint256, y: uint256) -> uint256: ...
    """
    
    contract = """
import base

resolutions: (
    base.process => self._process,
)

@internal
def _process(x: uint256) -> uint256:  # ERROR: Wrong signature
    return x * 2
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module})
        compile_code(contract, input_bundle=input_bundle)
    
    assert "Signature mismatch" in str(exc_info.value) or "signature" in str(exc_info.value).lower()


def test_multiple_modules_with_abstract_methods(make_input_bundle):
    """Test using multiple modules with abstract methods"""
    base_a_module = """
@abstract
def process_a(): ...

@external
def execute_a():
    self.process_a()
    """
    
    base_b_module = """
@abstract  
def process_b(): ...
    
@external
def execute_b():
    self.process_b()
    """
    
    contract = """
import base_a
import base_b

resolutions: (
    base_a.process_a => self._a,
    base_b.process_b  => self._b,
)

@internal
def _a():
    pass

@internal  
def _b():
    pass
    """
    
    # Should compile successfully - no restriction on multiple modules with abstracts
    input_bundle = make_input_bundle({"base_a.vy": base_a_module, "base_b.vy": base_b_module})
    compile_code(contract, input_bundle=input_bundle)


def test_unreachable_abstract_method(make_input_bundle):
    """Test that unreachable abstract methods don't need resolution"""
    base_module = """
@abstract
def used_method(): ...

@abstract
def unused_method(): ...

@external
def execute():
    self.used_method()
    # unused_method is never called
    """
    
    contract = """
import base

resolutions: (
    base.used_method => self._impl,
)

@internal
def _impl():
    pass

# No resolution for unused_method - should be OK since it's unreachable
    """
    
    # Should compile successfully (unreachable abstracts don't need resolution)
    input_bundle = make_input_bundle({"base.vy": base_module})
    compile_code(contract, input_bundle=input_bundle)


def test_fresh_implementation_no_provider(make_input_bundle):
    """Test providing fresh implementation when no provider exists"""
    base_module = """
@abstract
def process(): ...

@external
def execute():
    self.process()
    """
    
    contract = """
import base

resolutions: (
    base.process => self._my_process,  # Fresh implementation (no provider)
)

@internal
def _my_process():
    pass
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module})
    compile_code(contract, input_bundle=input_bundle)


def test_complex_resolution_chain(make_input_bundle):
    """Test complex chain of module dependencies and resolutions"""
    base_module = """
@abstract
def step1() -> uint256: ...

@abstract
def step2(x: uint256) -> uint256: ...

@abstract
def step3(x: uint256) -> uint256: ...

@external
def process() -> uint256:
    a: uint256 = self.step1()
    b: uint256 = self.step2(a)
    c: uint256 = self.step3(b)
    return c
    """
    
    partial_impl_module = """
import base

provides: (
        base.step1 => self._step1,
        base.step2 => self._step2,
    )

@internal
def _step1() -> uint256:
    return 10

@internal
def _step2(x: uint256) -> uint256:
    return x * 2
    """
    
    contract = """
import base
import partial

resolutions: (
            base.step1 => accept partial._step1,
            base.step2 => override self._my_step2[partial._step2],
            base.step3 => self._step3,
        )

@internal
def _my_step2(x: uint256) -> uint256:
    return x * 3  # Override with different logic

@internal
def _step3(x: uint256) -> uint256:
    return x + 100
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"base.vy": base_module, "partial.vy": partial_impl_module})
    compile_code(contract, input_bundle=input_bundle)


def test_abstract_external_function(make_input_bundle):
    """Test that abstract methods can be external"""
    module = """
@abstract
def callback(sender: address, amount: uint256): ...

@external
def execute(amount: uint256):
    self.callback(msg.sender, amount)
    """
    
    contract = """
import module

resolutions: (
            module.callback => self.handle_callback,
        )

@external
def handle_callback(sender: address, amount: uint256):
    pass
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"module.vy": module})
    compile_code(contract, input_bundle=input_bundle)


def test_abstract_view_function(make_input_bundle):
    """Test that abstract methods can be view functions"""
    module = """
@abstract
@view
def get_balance(account: address) -> uint256: ...

@external
@view
def total_balance(a: address, b: address) -> uint256:
    return self.get_balance(a) + self.get_balance(b)
    """
    
    contract = """
import module

resolutions: (
            module.get_balance => self._get_bal,
        )

balances: HashMap[address, uint256]

@internal
@view
def _get_bal(account: address) -> uint256:
    return self.balances[account]
    """
    
    # Should compile successfully
    input_bundle = make_input_bundle({"module.vy": module})
    compile_code(contract, input_bundle=input_bundle)


def test_provides_block_validation(make_input_bundle):
    """Test that provides block is validated properly"""
    base_module = """
@abstract
def foo(): ...
    """
    
    # Invalid: providing for non-existent abstract method
    invalid_module = """
import base

provides: (
        base.nonexistent => self._impl,  # ERROR: No such abstract method
    )

@internal
def _impl():
    pass
    """
    
    with pytest.raises(StructureException) as exc_info:
        input_bundle = make_input_bundle({"base.vy": base_module})
        compile_code(invalid_module, input_bundle=input_bundle)
    
    assert "nonexistent" in str(exc_info.value) or "No such abstract" in str(exc_info.value)
