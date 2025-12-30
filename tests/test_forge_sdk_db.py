import pathlib
import sys
import types
from importlib import import_module

import pytest

# Create a lightweight 'sdk' package pointing to the SDK source directory to avoid
# executing the heavy package __init__ that requires additional dependencies.
SDK_PATH = pathlib.Path(__file__).resolve().parent.parent / "backend/forge/forge/sdk"
sdk_pkg = types.ModuleType("sdk")
sdk_pkg.__path__ = [str(SDK_PATH)]
sys.modules.setdefault("sdk", sdk_pkg)

db_module = import_module("sdk.db")
errors_module = import_module("sdk.errors")
model_module = import_module("sdk.model")

AgentDB = db_module.AgentDB
NotFoundError = errors_module.NotFoundError
StepRequestBody = model_module.StepRequestBody
Status = model_module.Status


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / 'test.db'
    return AgentDB(f'sqlite:///{db_path}')


@pytest.mark.asyncio
async def test_create_and_query_entities(db):
    # create task
    task = await db.create_task('input')
    retrieved = await db.get_task(task.task_id)
    assert retrieved.task_id == task.task_id

    # create a step
    step_req = StepRequestBody(input='step')
    step = await db.create_step(task.task_id, step_req)
    fetched_step = await db.get_step(task.task_id, step.step_id)
    assert fetched_step.input == 'step'

    # update step and verify status/output
    await db.update_step(task.task_id, step.step_id, status='completed', output='done')
    updated_step = await db.get_step(task.task_id, step.step_id)
    assert updated_step.status == Status.completed
    assert updated_step.output == 'done'

    # create artifact and retrieve
    artifact = await db.create_artifact(task.task_id, 'file.txt', './')
    fetched_artifact = await db.get_artifact(artifact.artifact_id)
    assert fetched_artifact.file_name == 'file.txt'

    # listing helpers
    tasks, task_pagination = await db.list_tasks()
    assert len(tasks) == 1 and task_pagination.total_items == 1

    steps, _ = await db.list_steps(task.task_id)
    assert len(steps) == 1

    artifacts, _ = await db.list_artifacts(task.task_id)
    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_missing_entities_raise_not_found(db):
    with pytest.raises(NotFoundError):
        await db.get_task('missing')

    task = await db.create_task('input')
    with pytest.raises(NotFoundError):
        await db.get_step(task.task_id, 'missing')

    with pytest.raises(NotFoundError):
        await db.get_artifact('missing')
