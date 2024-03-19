import pytest

from tests.pyevm.pyevm_env import PyEVMEnv


@pytest.fixture(scope="module")
def pyevm():
    return PyEVMEnv(gas_limit=1e18)


@pytest.fixture(scope="module")
def get_contract(pyevm, optimize, output_formats):
    def fn(source_code, *args, **kwargs):
        return pyevm.deploy_source(source_code, optimize, output_formats, *args, **kwargs)

    return fn
