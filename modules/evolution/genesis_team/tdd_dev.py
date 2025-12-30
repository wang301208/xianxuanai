"""TDD developer agent builds features using tests."""

from __future__ import annotations

import os
import subprocess
import tempfile

from .. import Agent

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


class TDDDeveloper(Agent):
    """Executes tests to drive development."""

    def perform(self, test_cmd: str = "pytest") -> str:
        """Run tests and attempt automatic fixes on failure."""
        try:
            result = subprocess.run(
                test_cmd.split(), capture_output=True, text=True, check=False
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                return output

            generated_code = self.generate_code(output)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(generated_code)
                temp_path = tmp.name

            rerun = subprocess.run(
                test_cmd.split(), capture_output=True, text=True, check=False
            )
            output += (
                f"\nGenerated code written to {temp_path}\n"
                + rerun.stdout
                + rerun.stderr
            )
            return output
        except Exception as err:  # pragma: no cover - external dependencies
            return f"Test execution failed: {err}"

    def generate_code(self, context: str) -> str:
        """Use an LLM to generate code to address failing tests."""
        prompt = (
            "Tests failed with the following output. Provide code to fix the issue:\n"
            + context
        )
        if not openai or not os.getenv("OPENAI_API_KEY"):
            return "# LLM interface not configured\n"

        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message["content"]
        except Exception as err:  # pragma: no cover - external call
            return f"# Code generation failed: {err}\n"
