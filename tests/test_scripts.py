"""Every CLI in scripts/ must at least parse and expose a working --help.

A syntax error in a script is invisible to the rest of the suite, because nothing
imports scripts/ as a module -- one shipped in a commit before this test existed.
These checks are cheap and catch that class of breakage immediately.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no scripts found to check"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path: Path):
    """Catches unterminated strings, bad indentation and the like."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"{path.name} line {exc.lineno}: {exc.msg}")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs(path: Path):
    """--help must work: it is the contract these scripts are documented by."""
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{path.name} --help exited {result.returncode}\n{result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), f"{path.name} --help printed nothing"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_documents_its_arguments(path: Path):
    """Every argparse option needs help text -- a bare flag is not self-documenting."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if not any(str(f).startswith("-") for f in flags):
            continue
        if not any(kw.arg == "help" for kw in node.keywords):
            missing.append(str(flags[0]) if flags else "<unknown>")
    assert not missing, f"{path.name}: options without help text: {missing}"
