"""Commands to generate code using an LLM."""

import logging
from datetime import datetime

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.core.utils.json_schema import JSONSchema

COMMAND_CATEGORY = "code_generation"
COMMAND_CATEGORY_TITLE = "Code Generation"

logger = logging.getLogger(__name__)


@command(
    "generate_code",
    "Generate code from a prompt and save it to a file in the workspace",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Instructions for the code to generate",
            required=True,
        ),
    },
)
async def generate_code(prompt: str, agent: Agent) -> str:
    """Generate code using the configured LLM and write it to the workspace.

    Args:
        prompt: Instructions for the code to generate.

    Returns:
        str: The relative path to the file containing the generated code.
    """
    response = await agent.llm_provider.create_chat_completion(
        model_prompt=[ChatMessage.user(prompt)],
        functions=[],
        model_name=agent.llm.name,
    )
    code = response.response.content or ""
    filename = f"generated_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    await agent.workspace.write_file(filename, code)
    logger.info("Generated code written to %s", filename)
    return filename
