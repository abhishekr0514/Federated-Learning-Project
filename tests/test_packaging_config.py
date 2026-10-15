"""Verify setuptools package discovery configuration."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from setuptools import find_packages


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pyproject_limits_package_discovery():
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    find_cfg = data["tool"]["setuptools"]["packages"]["find"]
    assert "ml*" in find_cfg["include"]
    assert "backend*" in find_cfg["include"]
    assert "data*" in find_cfg["exclude"]
    assert "frontend*" in find_cfg["exclude"]


def test_find_packages_matches_intended_distribution():
    import os

    root = _repo_root()
    prev = os.getcwd()
    try:
        os.chdir(root)
        packages = find_packages(
            where=".",
            include=["backend*", "ml*", "privacy*", "fl*"],
            exclude=[
                "tests*",
                "data*",
                "frontend*",
                "deployment*",
                "docs*",
                "scripts*",
                "storage*",
            ],
        )
    finally:
        os.chdir(prev)
    top_level = {p.split(".")[0] for p in packages}
    assert top_level <= {"backend", "ml", "privacy", "fl"}
    assert "ml.ptbxl" in packages
    assert "data" not in top_level
    assert "frontend" not in top_level
