def test_self(w3, get_contract_with_gas_estimation):
    code = """
@external
def foo() -> address:
    return self
    """
    c = get_contract_with_gas_estimation(code)

    assert c.foo() == c.address
