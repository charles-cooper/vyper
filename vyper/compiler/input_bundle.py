import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterator, Optional

from vyper.exceptions import JSONError

# a type to make mypy happy
PathLike = Path | PurePath


class CompilerInput:
    # an input to the compiler.

    @staticmethod
    def from_string(source_id: int, path: PathLike, file_contents: str) -> "CompilerInput":
        try:
            s = json.loads(file_contents)
            return ABIInput(source_id, path, s)
        except (ValueError, TypeError):
            return FileInput(source_id, path, file_contents)


@dataclass
class FileInput(CompilerInput):
    source_id: int
    path: PathLike
    source_code: str


@dataclass
class ABIInput(CompilerInput):
    # some json input, that has already been parsed into a dict or list
    source_id: int
    path: PathLike
    abi: Any  # something that json.load() returns


class _NotFound(Exception):
    pass


# wrap os.path.normpath, but return the same type as the input
def _normpath(path):
    return path.__class__(os.path.normpath(path))


class InputBundle:
    search_paths: list[PathLike]
    # compilation_targets: dict[str, str]  # contract names => contract sources

    def __init__(self, search_paths):
        self.search_paths = search_paths
        self._source_id_counter = 0
        self._source_ids: dict[PathLike, int] = {}

    def _load_from_path(self, path):
        raise NotImplementedError(f"not implemented! {self.__class__}._load_from_path()")

    def _generate_source_id(self, path: PathLike) -> int:
        if path not in self._source_ids:
            self._source_ids[path] = self._source_id_counter
            self._source_id_counter += 1

        return self._source_ids[path]

    def load_file(self, path: PathLike | str) -> CompilerInput:
        # search path precedence
        tried = []
        for sp in reversed(self.search_paths):
            # note from pathlib docs:
            # > If the argument is an absolute path, the previous path is ignored.
            # Path("/a") / Path("/b") => Path("/b")
            to_try = sp / path

            # normalize the path with os.path.normpath, to break down
            # things like "foo/bar/../x.vy" => "foo/x.vy", with all
            # the caveats around symlinks that os.path.normpath comes with.
            to_try = _normpath(to_try)
            try:
                return self._load_from_path(to_try)
            except _NotFound:
                tried.append(to_try)

        formatted_search_paths = "\n".join(["  " + str(p) for p in tried])
        raise FileNotFoundError(
            f"could not find {path} in any of the following locations:\n"
            f"{formatted_search_paths}"
        )

    def add_search_path(self, path: PathLike) -> None:
        self.search_paths.append(path)

    # temporarily add something to the search path (within the
    # scope of the context manager) with highest precedence.
    # if `path` is None, do nothing
    @contextlib.contextmanager
    def search_path(self, path: Optional[PathLike]) -> Iterator[None]:
        if path is None:
            yield  # convenience, so caller does not have to handle null path

        else:
            self.search_paths.append(path)
            try:
                yield
            finally:
                self.search_paths.pop()


# regular input. takes a search path(s), and `load_file()` will search all
# search paths for the file and read it from the filesystem
class FilesystemInputBundle(InputBundle):
    def _load_from_path(self, path: Path) -> CompilerInput:
        try:
            with path.open() as f:
                code = f.read()
        except FileNotFoundError:
            raise _NotFound(path)

        source_id = super()._generate_source_id(path)

        return CompilerInput.from_string(source_id, path, code)


# fake filesystem for JSON inputs. takes a base path, and `load_file()`
# "reads" the file from the JSON input. Note that this input bundle type
# never actually interacts with the filesystem -- it is guaranteed to be pure!
class JSONInputBundle(InputBundle):
    input_json: dict[PurePath, Any]

    def __init__(self, input_json, search_paths):
        super().__init__(search_paths)
        self.input_json = {}
        for path, item in input_json.items():
            self.input_json[_normpath(path)] = item

    def _load_from_path(self, path: PurePath) -> CompilerInput:
        try:
            value = self.input_json[path]
        except KeyError:
            raise _NotFound(path)

        source_id = super()._generate_source_id(path)

        if "content" in value:
            return CompilerInput.from_string(source_id, path, value["content"])

        if "abi" in value:
            return ABIInput(source_id, path, value["abi"])

        # TODO: ethPM support
        # if isinstance(contents, dict) and "contractTypes" in contents:

        # unreachable, based on how JSONInputBundle is constructed in
        # the codebase.
        raise JSONError(f"Unexpected type in file: '{path}'")  # pragma: nocover
