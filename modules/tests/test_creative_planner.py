"""Tests for the CreativePlanner."""

import os
import sys
import logging
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

from third_party.autogpt.autogpt.core.planning.creative import CreativePlanner
from third_party.autogpt.autogpt.core.planning.simple import SimplePlanner
from third_party.autogpt.autogpt.core.planning.schema import TaskType
from third_party.autogpt.autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
)
from third_party.autogpt.autogpt.core.resource.model_providers.schema import (
    ChatModelProvider,
    ChatModelResponse,
    ModelProviderName, ModelProviderService, ChatModelInfo,
)


class FakeChatModelProvider(ChatModelProvider):
    def __init__(self, plans: list[dict[str, Any]]):
        self._plans = plans
        self._call_index = 0

    async def get_available_models(self) -> list[Any]:
        return []

    def count_message_tokens(self, messages: ChatMessage | list[ChatMessage], model_name: str) -> int:
        return 0

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser=lambda _: None,
        functions=None,
        max_output_tokens=None,
        **kwargs,
    ) -> ChatModelResponse:
        plan = self._plans[self._call_index]
        self._call_index += 1
        message = AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantToolCall(
                    id="1",
                    type="function",
                    function=AssistantFunctionCall(
                        name="create_initial_agent_plan",
                        arguments=plan,
                    ),
                )
            ],
        )
        parsed = completion_parser(message)
        return ChatModelResponse(response=message, parsed_result=parsed, prompt_tokens_used=0, completion_tokens_used=0, model_info=ChatModelInfo(name=model_name, provider_name=ModelProviderName.OPENAI, prompt_token_cost=0.0, completion_token_cost=0.0, max_tokens=1000))

    def count_tokens(self, text: str, model_name: str) -> int:
        return 0

    def get_tokenizer(self, model_name: str):
        class _T:
            def encode(self, text):
                return []

            def decode(self, tokens):
                return ""

        return _T()

    def get_token_limit(self, model_name: str) -> int:
        return 1000


@pytest.mark.asyncio
async def test_creative_planner_generates_multiple_plans():
    plans = [
        {
            "task_list": [
                {
                    "objective": "task one",
                    "type": TaskType.PLAN,
                    "priority": 1,
                    "ready_criteria": [],
                    "acceptance_criteria": [],
                }
            ]
        },
        {
            "task_list": [
                {
                    "objective": "task two",
                    "type": TaskType.PLAN,
                    "priority": 1,
                    "ready_criteria": [],
                    "acceptance_criteria": [],
                }
            ]
        },
    ]
    provider = FakeChatModelProvider(plans)
    planner = CreativePlanner(
        CreativePlanner.default_settings,
        logger=logging.getLogger("test"),
        model_providers={ModelProviderName.OPENAI: provider},
    )
    response = await planner.make_initial_plan(
        agent_name="Agent",
        agent_role="role",
        agent_goals=["goal"],
        abilities=["ability"],
        num_options=2,
    )
    assert len(response.parsed_result["plan_options"]) == 2


@pytest.mark.asyncio
async def test_creative_planner_is_planner():
    provider = FakeChatModelProvider([
        {
            "task_list": [
                {
                    "objective": "task one",
                    "type": TaskType.PLAN,
                    "priority": 1,
                    "ready_criteria": [],
                    "acceptance_criteria": [],
                }
            ]
        },
        {
            "task_list": [
                {
                    "objective": "task two",
                    "type": TaskType.PLAN,
                    "priority": 1,
                    "ready_criteria": [],
                    "acceptance_criteria": [],
                }
            ]
        },
    ])
    planner: SimplePlanner = CreativePlanner(
        CreativePlanner.default_settings,
        logger=logging.getLogger("test2"),
        model_providers={ModelProviderName.OPENAI: provider},
    )
    result = await planner.make_initial_plan(
        agent_name="Agent",
        agent_role="role",
        agent_goals=["goal"],
        abilities=["ability"],
        num_options=2,
    )
    assert "plan_options" in result.parsed_result
