"""Architecture guard: sub-packages may only depend on the layers below them.

Only runtime imports count; imports under `if TYPE_CHECKING:` are for annotations and are ignored.
Top-level modules (`bootstrap`, `cli`) are the wiring layer and may import anything.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import copilot

SRC = Path(copilot.__file__).resolve().parent

# package -> packages it may import at runtime (the dependency direction, low to high)
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "domain": {"core"},
    "data": {"core", "domain"},
    "retrieval": {"core", "domain"},
    "llm": {"core", "domain"},
    "observability": {"core", "domain"},
    "drafting": {"core", "domain", "retrieval"},
    "safety": {"core", "domain", "retrieval"},
    "agents": {"core", "domain", "drafting", "llm", "retrieval"},
    "workflow": {"core", "domain", "drafting", "observability", "retrieval", "safety"},
    "evaluation": {"core", "domain", "retrieval", "safety"},
}


def _package_of(path: Path) -> str | None:
    parts = path.relative_to(SRC).parts
    return parts[0] if len(parts) > 1 else None  # None = a top-level module


def _runtime_imports(path: Path) -> set[str]:
    """Names of `copilot` sub-packages imported at runtime by one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                continue
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        visit([child])
                continue
            parts = module.split(".")
            if parts[0] == "copilot" and len(parts) > 1:
                found.add(parts[1])

    visit(tree.body)
    return found


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if _package_of(p) is not None)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_packages_only_import_lower_layers(path: Path) -> None:
    package = _package_of(path)
    assert package in ALLOWED, f"unknown package '{package}': add it to ALLOWED"
    imported = _runtime_imports(path) - {package}
    # a sub-package may import top-level modules of nothing: bootstrap and cli sit above all
    forbidden = imported - ALLOWED[package]
    assert not forbidden, (
        f"{path.relative_to(SRC)} (layer '{package}') imports {sorted(forbidden)}; "
        f"it may only import {sorted(ALLOWED[package])}"
    )


def test_every_package_is_declared_and_has_an_init() -> None:
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert packages == set(ALLOWED)
    for name in packages:
        assert (SRC / name / "__init__.py").is_file()


def test_nothing_below_the_wiring_layer_imports_it() -> None:
    wiring = {"bootstrap", "cli"}
    offenders = [str(p.relative_to(SRC)) for p in _modules() if _runtime_imports(p) & wiring]
    assert offenders == []
