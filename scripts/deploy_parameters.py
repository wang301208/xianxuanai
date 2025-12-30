"""Apply parameter suggestions and monitor for regression."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import yaml


def apply_parameters(config_path: Path, params: dict) -> dict:
    config: dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
    config.update(params)
    config_path.write_text(yaml.safe_dump(config))
    return config


def run_tests(command: str) -> tuple[int, float, str]:
    """Run the regression tests securely.

    The command is validated against a small whitelist and executed without a
    shell. A non-zero exit code or validation error will be surfaced via the
    returned output message.
    """

    try:
        args = shlex.split(command)
    except ValueError as err:
        return 1, 0.0, f"Invalid test command: {err}"
    if not args:
        return 1, 0.0, "Empty test command"

    is_pytest = args[0] == "pytest"
    is_python_pytest = (
        args[0] == "python" and len(args) >= 3 and args[1] == "-m" and args[2] == "pytest"
    )
    if not (is_pytest or is_python_pytest):
        return 1, 0.0, f"Disallowed test command: {' '.join(args)}"

    start = time.perf_counter()
    try:
        proc = subprocess.run(args, shell=False, capture_output=True, text=True)
    except OSError as err:
        duration = time.perf_counter() - start
        return 1, duration, f"Failed to execute test command: {err}"

    duration = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        output = f"Test command exited with code {proc.returncode}:\n" + output
    return proc.returncode, duration, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy parameters and run regression tests"
    )
    parser.add_argument(
        "--analytics", default="analytics_output.json", help="Analytics JSON file"
    )
    parser.add_argument(
        "--config", default="prompt_settings.yaml", help="Config file to update"
    )
    parser.add_argument(
        "--test", default="pytest -q", help="Command used for regression tests"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.analytics).read_text())
    params = data.get("suggested_parameters", {})
    apply_parameters(Path(args.config), params)

    code, duration, output = run_tests(args.test)
    report = {"returncode": code, "duration": duration, "output": output}
    Path("deployment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
