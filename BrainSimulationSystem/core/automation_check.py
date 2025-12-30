"""Automated structural checks for the brain simulation codebase."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class CodeAnalyzer:
    """Collects structural issues for neural simulation modules."""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def analyze_module_structure(self) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []

        neuron_files = self._find_neuron_model_files()
        issues.extend(self._find_duplicate_models(neuron_files))
        issues.extend(self._check_interface_consistency(neuron_files))
        issues.extend(self._check_import_consistency())
        issues.extend(self._check_network_package_structure())

        return issues

    def _find_neuron_model_files(self) -> List[Path]:
        patterns = [
            "*neuron*.py",
            "*model*.py",
            "multi_neuron_models.py",
            "neuron_models.py",
            "neurons.py",
            "neuron_base.py",
        ]
        files: Set[Path] = set()
        for pattern in patterns:
            files.update(self.base_path.rglob(pattern))
        return sorted(files)

    def _find_duplicate_models(self, files: List[Path]) -> List[Dict[str, Any]]:
        model_defs: Dict[str, List[str]] = {}
        for path in files:
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in ast.walk(ast.parse(content)):
                if isinstance(match, ast.ClassDef) and match.name.endswith("Neuron"):
                    model_defs.setdefault(match.name, []).append(str(path))
        duplicates = {
            name: locations for name, locations in model_defs.items() if len(locations) > 1
        }
        if not duplicates:
            return []
        return [
            {
                "type": "duplicate-model",
                "severity": "high",
                "description": f"Neuron models defined multiple times: {duplicates}",
                "suggestion": "Consolidate duplicate neuron classes into a single module.",
            }
        ]

    def _check_interface_consistency(self, files: List[Path]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        expected_second = {"input_current", "inputs"}
        dt_aliases = {"dt", "delta_t", "time_step"}

        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue

            offenders: List[str] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or node.name != "update":
                    continue

                arg_names = [arg.arg for arg in node.args.args]
                if not arg_names:
                    continue
                if arg_names[0] != "self":
                    continue

                positional = arg_names[1:]
                if not positional:
                    continue

                first = positional[0]
                second = positional[1] if len(positional) > 1 else None

                if first in dt_aliases:
                    if second and second in expected_second:
                        continue
                    offenders.append(node.name)
                elif first not in expected_second:
                    offenders.append(node.name)

            if offenders:
                unique = sorted(set(offenders))
                issues.append(
                    {
                        "type": "inconsistent-interface",
                        "severity": "medium",
                        "description": f"{path}: update signatures missing expected input_current parameter ({unique}).",
                        "suggestion": "Normalize update(self, dt: float, input_current: float) across neuron models.",
                    }
                )

        return issues


    def _check_import_consistency(self) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        import_pattern = ast.ImportFrom
        imported: Dict[str, str] = {}

        for path in self.base_path.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, import_pattern) and node.names:
                    for alias in node.names:
                        key = alias.name
                        module = node.module or ""
                        if key in imported and imported[key] != module:
                            issues.append(
                                {
                                    "type": "import-inconsistency",
                                    "severity": "medium",
                                    "description": f"{path}: {key} imported from multiple modules.",
                                    "suggestion": f"Standardise imports of {key} to {imported[key]}",
                                }
                            )
                        else:
                            imported[key] = module
        return issues

    def _check_network_package_structure(self) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        core_dir = self.base_path / "core"
        legacy_module = core_dir / "network.py"
        if legacy_module.exists():
            issues.append(
                {
                    "type": "network-package",
                    "severity": "high",
                    "description": "Legacy core/network.py detected; the network should live in core/network/.",
                    "suggestion": "Remove core/network.py and rely on the package structure.",
                }
            )

        package_dir = core_dir / "network"
        if not package_dir.is_dir():
            issues.append(
                {
                    "type": "network-package",
                    "severity": "high",
                    "description": "Missing BrainSimulationSystem/core/network package.",
                    "suggestion": "Create the network package and split responsibilities across modules.",
                }
            )
            return issues

        required = {
            "__init__.py",
            "base.py",
            "dependencies.py",
            "full_brain.py",
            "initialization.py",
            "integration.py",
            "runtime.py",
        }
        missing = [name for name in required if not (package_dir / name).exists()]
        if missing:
            issues.append(
                {
                    "type": "network-package",
                    "severity": "medium",
                    "description": f"Network package is missing: {missing}",
                    "suggestion": "Ensure the network package exposes the expected modules.",
                }
            )

        max_lines = 1200
        oversized: List[Tuple[str, int]] = []
        for py_file in package_dir.rglob("*.py"):
            try:
                lines = sum(1 for _ in py_file.open(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            if lines > max_lines:
                oversized.append((str(py_file), lines))
        if oversized:
            issues.append(
                {
                    "type": "module-size",
                    "severity": "medium",
                    "description": f"Modules exceed {max_lines} lines: {oversized}",
                    "suggestion": "Split long modules into focused components.",
                }
            )

        class_locations: Dict[str, List[str]] = {}
        for py_file in package_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in {
                    "FullBrainNetwork",
                    "FullBrainNeuralNetwork",
                    "NeuralNetwork",
                }:
                    class_locations.setdefault(node.name, []).append(str(py_file))
        duplicates = {name: locations for name, locations in class_locations.items() if len(locations) > 1}
        if duplicates:
            issues.append(
                {
                    "type": "duplicate-definition",
                    "severity": "high",
                    "description": f"Core classes defined multiple times: {duplicates}",
                    "suggestion": "Keep one canonical definition per class and re-export via __init__.py.",
                }
            )

        return issues


def format_report(issues: List[Dict[str, Any]]) -> str:
    if not issues:
        return "✅  No structural issues detected."

    sections = {
        "high": [issue for issue in issues if issue["severity"] == "high"],
        "medium": [issue for issue in issues if issue["severity"] == "medium"],
        "low": [issue for issue in issues if issue["severity"] == "low"],
    }
    lines: List[str] = ["# Modularity review report"]
    labels = {
        "high": "## 🔴 High severity",
        "medium": "## 🟡 Medium severity",
        "low": "## 🟢 Low severity",
    }
    for level in ("high", "medium", "low"):
        bucket = sections[level]
        if not bucket:
            continue
        lines.append(labels[level])
        for issue in bucket:
            lines.append(f"### {issue['type']}")
            lines.append(f"- detail: {issue['description']}")
            lines.append(f"- suggestion: {issue['suggestion']}")
            lines.append("")
    return "\n".join(lines)


def run_automated_checks() -> List[Dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    analyzer = CodeAnalyzer(str(project_root))
    issues = analyzer.analyze_module_structure()
    report = format_report(issues)
    try:
        print(report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report.encode('utf-8', errors='ignore'))
        sys.stdout.buffer.write(b"\n")
    return issues


if __name__ == '__main__':
    run_automated_checks()
