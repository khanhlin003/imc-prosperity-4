from abc import abstractmethod
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import ContextManager, Optional


def default_data_root() -> Optional[Path]:
    """
    Prefer on-disk `data/` next to the `backtester/` package (repo/workspace layout),
    then `./data` from the current working directory.
    Returns None if neither exists.
    """
    pkg_dir = Path(__file__).resolve().parent
    # .../backtester/prosperity3bt/*.py -> parents[1] is the workspace root (parent of `backtester/`)
    repo_adjacent = pkg_dir.parents[1] / "data"
    if repo_adjacent.is_dir():
        return repo_adjacent
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data
    return None


@contextmanager
def wrap_in_context_manager(value):
    yield value


class FileReader:
    @abstractmethod
    def file(self, path_parts: list[str]) -> ContextManager[Optional[Path]]:
        """Given a path to a file, yields a single Path object to the file or None if the file does not exist."""
        raise NotImplementedError()


class FileSystemReader(FileReader):
    def __init__(self, root: Path) -> None:
        self._root = root

    def file(self, path_parts: list[str]) -> ContextManager[Optional[Path]]:
        file = self._root
        for part in path_parts:
            file = file / part

        if not file.is_file():
            return wrap_in_context_manager(None)

        return wrap_in_context_manager(file)


class PackageResourcesReader(FileReader):
    def file(self, path_parts: list[str]) -> ContextManager[Optional[Path]]:
        try:
            container = resources.files(f"prosperity3bt.resources.{'.'.join(path_parts[:-1])}")
            file = container / path_parts[-1]
            if not file.is_file():
                return wrap_in_context_manager(None)

            return resources.as_file(file)
        except Exception:
            return wrap_in_context_manager(None)
