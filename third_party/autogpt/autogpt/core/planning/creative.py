from autogpt.core.planning.simple import SimplePlanner
from autogpt.core.resource.model_providers import ChatModelResponse


class CreativePlanner(SimplePlanner):
    """Planner that generates multiple plan options using an LLM."""

    async def make_initial_plan(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
        num_options: int = 3,
    ) -> ChatModelResponse:
        plans = []
        last_response: ChatModelResponse | None = None
        for _ in range(num_options):
            last_response = await super().make_initial_plan(
                agent_name=agent_name,
                agent_role=agent_role,
                agent_goals=agent_goals,
                abilities=abilities,
            )
            plans.append(last_response.parsed_result)
        if last_response is None:
            raise RuntimeError("LLM did not return any plans")
        return ChatModelResponse(
            response=last_response.response,
            parsed_result={"plan_options": plans},
            prompt_tokens_used=last_response.prompt_tokens_used,
            completion_tokens_used=last_response.completion_tokens_used,
            model_info=last_response.model_info,
        )
