import asyncio
import logging
import re
import time
from functools import wraps
from pathlib import Path
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class _MetricsCollector:
    """Collect metrics and track success state."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self.success = True

    def add_metric(self, key: str, value: object) -> None:
        self._parts.append(f"{key}={value}")

    def add_error(self, key: str, error: object) -> None:
        self.success = False
        self._parts.append(f"{key}_error={error}")

    def fail(self) -> None:
        self.success = False

    def message(self) -> str:
        return ", ".join(self._parts)


def _handle_errors(metric_name: str, missing_msg: str | None = None):
    """Decorator to capture exceptions and record them in the collector."""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, collector: _MetricsCollector, *args, **kwargs):
            try:
                await func(self, collector, *args, **kwargs)
            except FileNotFoundError:
                collector.add_error(metric_name, missing_msg or "not installed")
            except Exception as e:  # pragma: no cover - best effort
                collector.add_error(metric_name, e)

        return wrapper

    return decorator


class EvaluateMetrics(Ability):
    """Evaluate code complexity and runtime for a Python file."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.EvaluateMetrics",
        ),
        workspace_required=True,
    )

    def __init__(self, logger: logging.Logger, workspace: Workspace) -> None:
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = (
        "Evaluate code metrics like cyclomatic complexity and execution time for a Python file."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relative path to the Python file to analyse.",
            required=True,
        )
    }

    async def __call__(self, file_path: str) -> AbilityResult:
        file_abs = self._workspace.get_path(file_path)
        collector = _MetricsCollector()
        try:
            source = file_abs.read_text()
        except Exception as e:  # pragma: no cover - best effort
            collector.add_error("file", e)
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"file_path": file_path},
                success=False,
                message=collector.message(),
            )

        await self._evaluate_complexity(collector, source)
        await self._measure_runtime(collector, file_abs)
        await self._collect_coverage(collector)
        await self._run_style_check(collector, file_abs)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"file_path": file_path},
            success=collector.success,
            message=collector.message(),
        )

    @_handle_errors("complexity")
    async def _evaluate_complexity(
        self, collector: _MetricsCollector, source: str
    ) -> None:
        from radon.complexity import cc_visit

        blocks = cc_visit(source)
        if blocks:
            avg_complexity = sum(b.complexity for b in blocks) / len(blocks)
        else:
            avg_complexity = 0.0
        collector.add_metric("complexity", f"{avg_complexity:.2f}")

    @_handle_errors("runtime")
    async def _measure_runtime(
        self, collector: _MetricsCollector, file_abs: Path
    ) -> None:
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(file_abs),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        runtime = time.perf_counter() - start
        collector.add_metric("runtime", f"{runtime:.4f}")
        if proc.returncode != 0:
            collector.fail()

    @_handle_errors("coverage", "pytest not installed")
    async def _collect_coverage(self, collector: _MetricsCollector) -> None:
        proc = await asyncio.create_subprocess_exec(
            "pytest",
            "--cov",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self._workspace.root),
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if match:
            collector.add_metric("coverage", f"{match.group(1)}%")
        else:
            collector.add_error("coverage", "unparsed")
        if proc.returncode != 0:
            collector.fail()

    @_handle_errors("style", "ruff not installed")
    async def _run_style_check(
        self, collector: _MetricsCollector, file_abs: Path
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            "ruff",
            "check",
            str(file_abs),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode().strip()
        if output:
            error_count = len(output.splitlines())
            collector.add_metric("style_errors", error_count)
            collector.fail()
        else:
            collector.add_metric("style_errors", 0)
        if proc.returncode not in (0, 1):
            collector.fail()
