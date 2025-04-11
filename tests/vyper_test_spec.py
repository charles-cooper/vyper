from _pytest.fixtures import FixtureRequest
import pytest
import yaml

class VyperFile(pytest.File):
    def collect(self):
        filepath = self.path
        yield VyperTestItem.from_parent(self, name=filepath.stem)


class DummyFixtureRequest(FixtureRequest):
    def __init__(self, node):
        super().__init__(node)

        self._node = node

    def _check_scope(self, fixturedef):
        # minimal implementation: assume all fixtures are valid
        return True
    @property
    def _scope(self):
        # assume function scoped fixtures
        return "function"
    def addfinalizer(self, finalizer):
        # Optionally collect finalizers on the node for later execution
        if not hasattr(self.node, "_finalizers"):
            self.node._finalizers = []
        self.node._finalizers.append(finalizer)

class VyperTestItem(pytest.Item):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        # The fixture system requires this dict to be present.
        self.funcargs = {}

    def runtest(self):
        # load the get_contract fixture dynamically
        #get_contract = self.session._fixturemanager.getfixturevalue("get_contract", self)
        req = DummyFixtureRequest(self)
        get_contract = req.getfixturevalue("get_contract")

        with open(self.path) as f:
            contents = f.open()

        lines = contents.splitlines(keepends=True)

        divider = None
        for i in range(len(lines)):
            line = lines[i]
            if "------" in line:
                divider = i
                break

        vyper_part = "".join(lines[:divider])

        yaml_part = "".join(lines[divider + 1:])

        spec = yaml.load_safe(yaml_part)

        contract = get_contract(vyper_part)

        for fn_name, cases in spec["cases"].values():
            input_ = case["input"]
            expected_output = case["output"]

            fn = getattr(contract, fn_name)
            if (actual := fn(*input_)) != expected_output:
                self.expected = expected_output
                self.actual = actual
                raise VyperTestException(self, self.name)

        if hasattr(self, "_finalizers"):
            for final in self._finalizers:
                final()

    def __repr_failure(self, excinfo):
        if isinstance(excinfo.value, VyperTestException):
            return (
                f"Test failed : {self.name}\n"
                f"Expected: {self.expected}\n"
                f"Actual: {self.actual}\n"
            )

    def repr_failure(self, excinfo):
        # Use the built-in traceback
        return self._repr_failure_py(excinfo, style="long")

    def reportinfo(self):
        return self.fspath, 0, f"Vyper test file: {self.name}"


class VyperTestException(Exception):
    pass
