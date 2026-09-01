import ast
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAT_BUNDLEADJUST_REPO = Path(
    os.environ.get("SAT_BUNDLEADJUST_REPO", "/home/roger/sat-bundleadjust")
).expanduser()
CORE_PACKAGE = SAT_BUNDLEADJUST_REPO / "bundle_adjust"
FORBIDDEN_ROOTS = {"eval_utils", "notebooks"}


def _iter_python_files(root):
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _import_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".", 1)[0]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value.split(".", 1)[0]


def _string_literals(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def test_bundle_adjust_does_not_depend_on_eval_or_notebooks():
    assert CORE_PACKAGE.is_dir(), (
        "Could not find sat-bundleadjust's bundle_adjust package. "
        "Set SAT_BUNDLEADJUST_REPO=/path/to/sat-bundleadjust."
    )

    offenders = []

    for path in _iter_python_files(CORE_PACKAGE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for root in _import_roots(tree):
            if root in FORBIDDEN_ROOTS:
                offenders.append(
                    f"{path.relative_to(SAT_BUNDLEADJUST_REPO)} imports {root}"
                )

        for value in _string_literals(tree):
            normalized = value.replace("\\", "/")
            for root in FORBIDDEN_ROOTS:
                if root in normalized.split("/"):
                    offenders.append(
                        f"{path.relative_to(SAT_BUNDLEADJUST_REPO)} references {root}"
                    )

    assert not offenders, "\n".join(offenders)
