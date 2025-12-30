"""Utilities for constructing agents from blueprint files.

This module parses agent blueprint YAML files and instantiates fully
configured :class:`autogpt.agents.agent.Agent` objects. Blueprints
specify the core prompt for the agent as well as which tools it is
permitted to use. The factory uses AutoGPT's configuration and capability
system to ensure the created agent only has access to the specified
tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import ast
import asyncio
import hashlib
import logging
from functools import lru_cache
from json import JSONDecodeError
from datetime import datetime
from uuid import uuid4

import aiofiles
import yaml
from jsonschema import validate

from third_party.autogpt.autogpt.agents.agent import Agent
from third_party.autogpt.autogpt.agent_factory.configurators import create_agent
from third_party.autogpt.autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
from third_party.autogpt.autogpt.config import AIProfile, Config
from third_party.autogpt.autogpt.config.ai_directives import AIDirectives
from third_party.autogpt.autogpt.core.brain.config import BrainBackend
from third_party.autogpt.autogpt.core.resource.model_providers import ChatModelProvider
from third_party.autogpt.autogpt.file_storage.base import FileStorage
from third_party.autogpt.autogpt.models.command_registry import CommandRegistry
from forge.sdk.model import Task

from third_party.autogpt.autogpt.core.errors import (
    AutoGPTError,
    SkillSecurityError,
)

from capability.librarian import Librarian
from org_charter import io as charter_io
from backend.monitoring.global_workspace import global_workspace
from knowledge import UnifiedKnowledgeBase
from reasoning import DecisionEngine
from third_party.autogpt.autogpt.core.brain.transformer_brain import TransformerBrain
from modules.brain.backends import BrainBackendInitError, create_brain_backend

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from backend.execution.conductor import AgentConductor
    from backend.world_model import WorldModel


logger = logging.getLogger(__name__)
SAFE_BUILTINS: dict[str, object] = {
    "__import__": __import__,
    "len": len,
    "range": range,
    "print": print,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
}


def _verify_skill(name: str, code: str, metadata: dict) -> None:
    """Ensure skill source passes signature verification."""

    signature = metadata.get("signature")
    if not signature:
        raise SkillSecurityError(name, "missing signature")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if signature != digest:
        raise SkillSecurityError(name, "invalid signature")


@lru_cache
def _parse_blueprint(path: Path) -> dict:
    """Load and validate a blueprint file.

    This function asynchronously reads the blueprint from disk and caches the
    parsed result so subsequent calls for the same path avoid disk I/O.

    Parameters
    ----------
    path: Path
        Path to a YAML file conforming to ``schemas/agent_blueprint.yaml``.
    """

    async def _read() -> str:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    try:
        content = asyncio.run(_read())
    except RuntimeError:
        # If an event loop is already running, fall back to creating a new one
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            content = loop.run_until_complete(_read())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    data = yaml.safe_load(content)
    validate(instance=data, schema=charter_io.BLUEPRINT_SCHEMA)
    return data


def _load_additional_tool(
    name: str, registry: CommandRegistry, repo_path: Path
) -> None:
    """Load a tool from the skill library and register it.

    This uses the :mod:`capability` package which serves as a very light-weight
    plugin system.  Skills are stored as Python source files and are expected to
    expose callables decorated with :func:`autogpt.command_decorator.command`.
    """
    librarian = Librarian(str(repo_path))
    try:
        code, meta = librarian.get_skill(name)
    except (OSError, PermissionError, JSONDecodeError) as err:
        logger.exception("Failed to retrieve tool '%s' from skill library", name)
        raise AutoGPTError(f"Failed to load tool '{name}'") from err

    try:
        _verify_skill(name, code, meta)
    except SkillSecurityError as err:
        logger.exception("Security violation for skill %s: %s", name, err.cause)
        raise

    namespace: dict[str, object] = {}
    try:
        parsed = ast.parse(code, mode="exec")
        exec(compile(parsed, filename=name, mode="exec"), {"__builtins__": SAFE_BUILTINS}, namespace)
    except (SyntaxError, TypeError, ValueError) as err:
        logger.exception("Error executing tool '%s'", name)
        raise AutoGPTError(f"Failed to initialize tool '{name}'") from err

    for obj in namespace.values():
        if getattr(obj, AUTO_GPT_COMMAND_IDENTIFIER, False):
            cmd = getattr(obj, "command", None)
            if cmd:
                registry.register(cmd)
                break


def _filter_authorized_tools(registry: CommandRegistry, allowed: Iterable[str]) -> None:
    allowed = set(allowed)
    for name, cmd in list(registry.commands.items()):
        if name not in allowed:
            registry.unregister(cmd)


def create_agent_from_blueprint(
    blueprint_path: Path | str,
    config: Config,
    llm_provider: ChatModelProvider,
    file_storage: FileStorage,
    world_model: "WorldModel | None" = None,
    conductor: "AgentConductor | None" = None,
) -> Agent:
    """Instantiate an :class:`Agent` from a blueprint file.

    Parameters
    ----------
    blueprint_path:
        Path to a YAML blueprint file describing the agent.
    config:
        Application configuration to use for the new agent.
    llm_provider:
        Model provider used for prompting.
    file_storage:
        File storage backend for the agent.
    world_model:
        Optional world model used to record visual observations produced by
        the agent during execution.
    """
    path = Path(blueprint_path)
    blueprint = _parse_blueprint(path)

    profile = AIProfile(
        ai_name=blueprint["role_name"],
        ai_role=blueprint["role_name"],
        ai_goals=[blueprint["core_prompt"]],
    )

    directives = AIDirectives.from_file(config.prompt_settings_file)
    backend_value = blueprint.get("brain_backend")
    selected_backend = config.brain_backend
    if backend_value is not None:
        try:
            selected_backend = BrainBackend(str(backend_value).lower())
        except ValueError:
            logger.warning(
                "Unsupported brain backend '%s' in blueprint; using %s",
                backend_value,
                selected_backend.value,
            )

    enable_brain = blueprint.get("use_transformer_brain", config.use_transformer_brain)
    enable_kb = blueprint.get("use_knowledge_base", config.use_knowledge_base)
    enable_de = blueprint.get("use_decision_engine", config.use_decision_engine)

    brain = (
        TransformerBrain()
        if enable_brain and selected_backend == BrainBackend.TRANSFORMER
        else None
    )
    whole_brain = None
    if selected_backend in (BrainBackend.WHOLE_BRAIN, BrainBackend.BRAIN_SIMULATION):
        try:
            whole_brain = create_brain_backend(
                selected_backend,
                whole_brain_config=config.whole_brain,
                brain_simulation_config=config.brain_simulation,
            )
        except BrainBackendInitError as exc:
            logger.warning(
                "Unable to initialise %s backend for agent '%s': %s",
                selected_backend.value,
                blueprint.get("role_name", "unknown"),
                exc,
            )
            whole_brain = None
    knowledge_base = UnifiedKnowledgeBase() if enable_kb else None
    if knowledge_base is not None:
        try:
            from backend.introspection.self_knowledge import bootstrap_self_knowledge
        except Exception:  # pragma: no cover - optional module
            bootstrap_self_knowledge = None  # type: ignore[assignment]

        registry = None
        if bootstrap_self_knowledge is not None:
            try:
                from capability import get_skill_registry  # type: ignore

                if callable(get_skill_registry):
                    registry = get_skill_registry()
            except Exception:
                registry = None
            try:
                bootstrap_self_knowledge(  # type: ignore[misc]
                    repo_root=Path.cwd(),
                    knowledge_base=knowledge_base,
                    registry=registry,
                    graph_store=getattr(knowledge_base, "graph_store", None),
                )
            except Exception:
                logger.debug("Self-knowledge bootstrap failed.", exc_info=True)
    decision_engine = None
    agent_identifier = blueprint["role_name"]
    if conductor is not None:
        try:
            decision_engine = conductor.create_decision_engine(agent_identifier)
        except Exception:
            logger.debug("Conductor failed to allocate decision engine", exc_info=True)
            decision_engine = DecisionEngine()
    elif enable_de:
        decision_engine = DecisionEngine()

    task_model = Task(
        input=blueprint["core_prompt"],
        additional_input=None,
        created_at=datetime.now(),
        modified_at=datetime.now(),
        task_id=str(uuid4()),
        artifacts=[],
    )

    try:
        agent = create_agent(
            agent_id=agent_identifier,
            task=task_model,
            ai_profile=profile,
            directives=directives,
            app_config=config,
            file_storage=file_storage,
            llm_provider=llm_provider,
            brain=brain,
            whole_brain=whole_brain,
            knowledge_base=knowledge_base,
            decision_engine=decision_engine,
        )
    except Exception:
        if conductor is not None:
            conductor.unregister_agent(agent_identifier)
        raise

    if conductor is not None:
        try:
            conductor.register_agent(agent)
        except Exception:
            logger.debug("Failed to register agent with conductor", exc_info=True)

    if world_model is not None:
        def record_visual(image=None, features=None, *, _wm=world_model, _aid=agent.settings.agent_id):
            _wm.add_visual_observation(_aid, image=image, features=features)

        setattr(agent, "record_visual_observation", record_visual)
        teacher = getattr(agent, "_self_teacher", None)
        if teacher is not None:
            attach = getattr(teacher, "attach_world_model", None)
            if callable(attach):
                try:
                    attach(world_model=world_model)
                except TypeError:
                    attach(world_model)  # type: ignore[misc]

    # Register core components with the global workspace so they can
    # exchange state and attention with other modules.
    global_workspace.register_module(agent.settings.agent_id, agent)
    command_registry_key = f"{agent.settings.agent_id}.command_registry"
    llm_provider_key = f"{agent.settings.agent_id}.llm_provider"
    global_workspace.register_module(command_registry_key, agent.command_registry)
    global_workspace.register_module(llm_provider_key, agent.llm_provider)
    setattr(
        agent,
        "workspace_keys",
        (
            agent.settings.agent_id,
            command_registry_key,
            llm_provider_key,
        ),
    )

    # Restrict commands to authorised tools
    _filter_authorized_tools(
        agent.command_registry, blueprint.get("authorized_tools", [])
    )

    # Attempt to load additional tools from the capability library if they were
    # requested but not part of the default command set.
    repo_root = Path.cwd()
    for tool_name in blueprint.get("authorized_tools", []):
        if tool_name not in agent.command_registry.commands:
            _load_additional_tool(tool_name, agent.command_registry, repo_root)

    # Store subscribed topics for potential later use by messaging layers.
    setattr(agent, "subscribed_topics", blueprint.get("subscribed_topics", []))
    return agent


__all__ = ["create_agent_from_blueprint"]
