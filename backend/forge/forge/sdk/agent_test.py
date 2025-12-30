import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

# Provide lightweight stubs for optional dependencies used during module import.
if "litellm" not in sys.modules:  # pragma: no cover - import guard
    async def _fake_acompletion(**kwargs):
        return None

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.AuthenticationError = Exception
    fake_litellm.InvalidRequestError = Exception
    fake_litellm.ModelResponse = object
    fake_litellm.acompletion = _fake_acompletion
    sys.modules["litellm"] = fake_litellm

if "openai" not in sys.modules:  # pragma: no cover - import guard
    fake_openai = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - minimal stub
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    return None

        class embeddings:
            @staticmethod
            def create(**kwargs):
                return None

    fake_openai.OpenAI = OpenAI
    sys.modules["openai"] = fake_openai

    fake_types = types.ModuleType("openai.types")
    class CreateEmbeddingResponse:  # pragma: no cover - stub
        pass
    fake_types.CreateEmbeddingResponse = CreateEmbeddingResponse
    sys.modules["openai.types"] = fake_types

    fake_audio_types = types.ModuleType("openai.types.audio")
    class Transcription:  # pragma: no cover - stub
        pass
    fake_audio_types.Transcription = Transcription
    sys.modules["openai.types.audio"] = fake_audio_types

if "modules.skills" not in sys.modules:  # pragma: no cover - import guard
    skills_pkg = types.ModuleType("modules.skills")
    skills_pkg.__path__ = []
    sys.modules["modules.skills"] = skills_pkg

if "modules.skills.executor" not in sys.modules:  # pragma: no cover - import guard
    fake_executor = types.ModuleType("modules.skills.executor")

    class SkillSandbox:  # pragma: no cover - minimal stub
        def __init__(self, *args, **kwargs):
            pass

        def run(self, func, payload, metadata=None):
            return func(payload)

    fake_executor.SkillSandbox = SkillSandbox
    sys.modules["modules.skills.executor"] = fake_executor

@pytest.fixture()
def agent(tmp_path):
    from forge.agent import ForgeAgent
    from forge.sdk import AgentDB, LocalWorkspace
    from forge.sdk.model import StepRequestBody, TaskRequestBody

    db_path = tmp_path / "test_db.sqlite3"
    workspace_root = tmp_path / "workspace"
    db = AgentDB(f"sqlite:///{db_path}")
    workspace = LocalWorkspace(str(workspace_root))
    return ForgeAgent(db, workspace)


@pytest.mark.anyio
async def test_execute_step_runs_reasoning_pipeline(agent):
    from forge.sdk.model import StepRequestBody, TaskRequestBody

    task = await agent.create_task(
        TaskRequestBody(input="Identify the capital of France", additional_input={})
    )

    step_request = StepRequestBody(
        name="analysis",
        input="Provide the answer with context",
    )

    step = await agent.execute_step(task.task_id, step_request)

    assert step.output is not None
    assert "Plan:" in step.output
    assert "Result:" in step.output
    assert "Washington D.C" not in step.output
    assert step.is_last is True

    artifacts = agent.workspace.list(task.task_id, "outputs")
    assert artifacts, "Step output artifact should be persisted"
    saved_output = agent.workspace.read(task.task_id, artifacts[0]).decode("utf-8")
    assert saved_output == step.output


@pytest.mark.anyio
async def test_execute_step_supports_continuations(agent):
    from forge.sdk.model import StepRequestBody, TaskRequestBody

    task = await agent.create_task(
        TaskRequestBody(input="Draft a multi-part report", additional_input={})
    )

    step_request = StepRequestBody(
        name="follow up",
        input="Create an outline",
        additional_input={"continue": True},
    )

    step = await agent.execute_step(task.task_id, step_request)

    assert step.is_last is False
    assert step.additional_output is not None
    assert "follow_up" in step.additional_output
    assert "Next:" in step.output
    assert step.status.value == "completed"
