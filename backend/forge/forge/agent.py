from typing import Any, Dict, Tuple

try:  # Prefer installed autogpt-forge package when available
    from forge.actions import ActionRegister  # type: ignore
    from forge.sdk import (  # type: ignore
        Agent,
        AgentDB,
        ForgeLogger,
        Step,
        StepRequestBody,
        Task,
        TaskRequestBody,
        Workspace,
    )
except ModuleNotFoundError:  # Fallback to monorepo package layout
    from .actions import ActionRegister
    from .sdk import (
        Agent,
        AgentDB,
        ForgeLogger,
        Step,
        StepRequestBody,
        Task,
        TaskRequestBody,
        Workspace,
    )

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.abilities = ActionRegister(self)

    def _plan_action(
        self,
        task: Task,
        step_request: StepRequestBody,
        wants_continuation: bool,
    ) -> Tuple[str, str, Dict[str, Any], str]:
        """
        Produce a lightweight plan for the step and pick an action to execute.

        This routine intentionally keeps reasoning simple to avoid external dependencies
        while still grounding decisions in the task and step context.
        """

        plan_summary = " ".join(
            filter(
                None,
                [
                    f"Goal: {task.input}",
                    f"Step: {step_request.input}" if step_request.input else None,
                    f"Additional input: {step_request.additional_input}"
                    if step_request.additional_input
                    else None,
                ],
            )
        )

        available_actions = list(self.abilities.abilities.keys())
        if not available_actions:
            raise RuntimeError("No registered actions available for execution")

        requested_action = None
        if step_request.additional_input:
            requested_action = step_request.additional_input.get("action")

        selected_action: str | None = None
        action_reason = ""

        if requested_action and requested_action in self.abilities.abilities:
            selected_action = requested_action
            action_reason = "Requested action provided in step input"
        elif "finish" in self.abilities.abilities and not wants_continuation:
            selected_action = "finish"
            action_reason = "Defaulted to finishing the task for a complete response"
        else:
            selected_action = available_actions[0]
            action_reason = "Selected the first available action as a fallback"

        action_args: Dict[str, Any] = {}
        if selected_action == "finish":
            action_args["reason"] = (
                step_request.input
                or task.input
                or "Completed planning without additional input"
            )
        elif step_request.additional_input:
            provided_args = step_request.additional_input.get("action_input") or {}
            if isinstance(provided_args, dict):
                action_args.update(provided_args)

        return plan_summary, selected_action, action_args, action_reason

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        LOG.info("🚀 Executing step for task %s", task_id)

        try:
            task = await self.db.get_task(task_id)
        except Exception as err:  # pragma: no cover - defensive guard
            LOG.error("Failed to load task context", exc_info=err)
            raise

        wants_continuation = bool(
            step_request.additional_input and step_request.additional_input.get("continue")
        )

        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            is_last=not wants_continuation,
            additional_input=step_request.additional_input or {},
        )

        await self.db.update_step(task_id, step.step_id, status="running")

        plan_summary, selected_action, action_args, action_reason = self._plan_action(
            task, step_request, wants_continuation
        )

        action_result: str | None = None
        try:
            if selected_action not in self.abilities.abilities:
                raise ValueError(f"Action '{selected_action}' is not registered")

            raw_action_result = await self.abilities.run_action(
                task_id, selected_action, **action_args
            )
            action_result = (
                raw_action_result
                if isinstance(raw_action_result, str)
                else repr(raw_action_result)
            )
        except Exception as err:  # pragma: no cover - defensive guard
            LOG.error("Action invocation failed", exc_info=err)
            action_result = f"Action failed: {err}"

        follow_up_prompt = (
            "Provide additional details to continue refining the solution."
            if wants_continuation
            else None
        )

        output_text = "\n".join(
            filter(
                None,
                [
                    f"Plan: {plan_summary}",
                    f"Action: {selected_action} ({action_reason})",
                    f"Result: {action_result}",
                    f"Next: {follow_up_prompt}" if follow_up_prompt else None,
                ],
            )
        )

        safe_name = (step_request.name or step_request.input or "step").strip().replace(
            " ", "_"
        )
        artifact_path = f"outputs/{safe_name or 'step'}_{step.step_id[:8]}.txt"
        try:
            self.workspace.write(
                task_id=task_id,
                path=artifact_path,
                data=output_text.encode("utf-8"),
            )

            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=artifact_path.split("/")[-1],
                relative_path="outputs",
                agent_created=True,
            )
        except Exception as err:  # pragma: no cover - defensive guard
            LOG.error("Failed to persist step outputs", exc_info=err)

        additional_output = {
            "plan": plan_summary,
            "action": selected_action,
            "action_result": action_result,
            "action_reason": action_reason,
        }
        if follow_up_prompt:
            additional_output["follow_up"] = follow_up_prompt

        completed_step = await self.db.update_step(
            task_id,
            step.step_id,
            status="completed",
            output=output_text,
            additional_output=additional_output,
        )

        LOG.info(
            "\t✅ Step %s completed with status %s", completed_step.step_id, completed_step.status
        )

        return completed_step
