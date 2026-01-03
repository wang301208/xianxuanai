"""Main script for the autogpt package."""
import logging
import os
from logging import _nameToLevel as logLevelMap
from pathlib import Path
from typing import Optional

import click
from pydantic import ValidationError

from autogpt.config.validation import validate_env

from autogpt.logs.config import LogFormatName

from .telemetry import setup_telemetry


@click.group(invoke_without_command=True)
@click.option(
    "--brain-backend",
    type=click.Choice(
        ["brain_simulation", "whole_brain", "transformer", "llm"],
        case_sensitive=False,
    ),
    default=None,
    help="Override cognitive backend (default: brain_simulation). Choices: llm, transformer, whole_brain, brain_simulation.",
)
@click.pass_context
def cli(ctx: click.Context, brain_backend: str | None):
    if brain_backend:
        os.environ["BRAIN_BACKEND"] = str(brain_backend).lower()
    setup_telemetry()
    try:
        validate_env()
    except ValidationError as err:
        missing = ", ".join(e["loc"][0] for e in err.errors())
        click.echo(
            f"Missing required environment variables: {missing}", err=True
        )
        raise SystemExit(1) from err

    # Invoke `run` by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option(
    "--autonomous",
    is_flag=True,
    help="Run without requiring user authorization for each action",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
@click.option(
    # TODO: this is a hidden option for now, necessary for integration testing.
    # We should make this public once we're ready to roll out agent specific workspaces.
    "--workspace-directory",
    "-w",
    type=click.Path(file_okay=False),
    hidden=True,
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-settings",
    "-C",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Specifies which ai_settings.yaml file to use, relative to AutoGPT's"
        " config directory. Will also automatically skip the re-prompt."
    ),
)
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Specifies which prompt_settings.yaml file to use, relative to AutoGPT's"
        " config directory."
    ),
)
@click.option(
    "--constraint",
    type=str,
    multiple=True,
    help=(
        "Add or override AI constraints to include in the prompt;"
        " may be used multiple times to pass multiple constraints"
    ),
)
@click.option(
    "--resource",
    type=str,
    multiple=True,
    help=(
        "Add or override AI resources to include in the prompt;"
        " may be used multiple times to pass multiple resources"
    ),
)
@click.option(
    "--best-practice",
    type=str,
    multiple=True,
    help=(
        "Add or override AI best practices to include in the prompt;"
        " may be used multiple times to pass multiple best practices"
    ),
)
@click.option(
    "--override-directives",
    is_flag=True,
    help=(
        "If specified, --constraint, --resource and --best-practice will override"
        " the AI's directives instead of being appended to them"
    ),
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
def run(
    continuous: bool,
    continuous_limit: Optional[int],
    autonomous: bool,
    speak: bool,
    gpt3only: bool,
    gpt4only: bool,
    browser_name: Optional[str],
    allow_downloads: bool,
    workspace_directory: Optional[Path],
    install_plugin_deps: bool,
    skip_news: bool,
    skip_reprompt: bool,
    ai_settings: Optional[Path],
    ai_name: Optional[str],
    ai_role: Optional[str],
    prompt_settings: Optional[Path],
    resource: tuple[str],
    constraint: tuple[str],
    best_practice: tuple[str],
    override_directives: bool,
    debug: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    log_file_format: Optional[str],
) -> None:
    """
    Sets up and runs an agent, based on the task specified by the user, or resumes an
    existing agent.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    if autonomous:
        from autogpt.runner.auto_loop import run_auto_loop

        run_auto_loop(
            continuous_limit=continuous_limit,
            ai_settings=ai_settings,
            prompt_settings=prompt_settings,
            speak=speak,
            debug=debug,
            log_level=log_level,
            log_format=log_format,
            log_file_format=log_file_format,
            gpt3only=gpt3only,
            gpt4only=gpt4only,
            browser_name=browser_name,
            allow_downloads=allow_downloads,
            skip_news=skip_news,
            workspace_directory=workspace_directory,
            install_plugin_deps=install_plugin_deps,
            override_ai_name=ai_name,
            override_ai_role=ai_role,
            resources=list(resource),
            constraints=list(constraint),
            best_practices=list(best_practice),
            override_directives=override_directives,
        )
    else:
        from autogpt.app.main import run_auto_gpt

        run_auto_gpt(
            continuous=continuous,
            continuous_limit=continuous_limit,
            ai_settings=ai_settings,
            prompt_settings=prompt_settings,
            skip_reprompt=skip_reprompt,
            speak=speak,
            debug=debug,
            log_level=log_level,
            log_format=log_format,
            log_file_format=log_file_format,
            gpt3only=gpt3only,
            gpt4only=gpt4only,
            browser_name=browser_name,
            allow_downloads=allow_downloads,
            skip_news=skip_news,
            workspace_directory=workspace_directory,
            install_plugin_deps=install_plugin_deps,
            override_ai_name=ai_name,
            override_ai_role=ai_role,
            resources=list(resource),
            constraints=list(constraint),
            best_practices=list(best_practice),
            override_directives=override_directives,
        )


@cli.command()
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Specifies which prompt_settings.yaml file to use, relative to AutoGPT's"
        " config directory."
    ),
)
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
def serve(
    prompt_settings: Optional[Path],
    gpt3only: bool,
    gpt4only: bool,
    browser_name: Optional[str],
    allow_downloads: bool,
    install_plugin_deps: bool,
    debug: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    log_file_format: Optional[str],
) -> None:
    """
    Starts an Agent Protocol compliant AutoGPT server, which creates a custom agent for
    every task.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.app.main import run_auto_gpt_server

    run_auto_gpt_server(
        prompt_settings=prompt_settings,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        install_plugin_deps=install_plugin_deps,
    )


@cli.command()
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Specifies which prompt_settings.yaml file to use, relative to AutoGPT's"
        " config directory."
    ),
)
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
def mcp(
    prompt_settings: Path | None,
    gpt3only: bool,
    gpt4only: bool,
    browser_name: str | None,
    allow_downloads: bool,
    install_plugin_deps: bool,
    debug: bool,
    log_level: str | None,
    log_format: str | None,
    log_file_format: str | None,
) -> None:
    """Starts a Model Context Protocol (MCP) bridge for AutoGPT over stdio."""
    from autogpt.app.mcp_server import run_auto_gpt_mcp_server

    run_auto_gpt_mcp_server(
        prompt_settings=prompt_settings,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        install_plugin_deps=install_plugin_deps,
    )


@cli.command()
@click.option(
    "--scenarios-dir",
    type=click.Path(path_type=Path),
    help="Directory containing replay scenario JSON files.",
)
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path),
    help="Directory where replay reports will be written.",
)
@click.option("--verbose", is_flag=True, help="Print detailed replay results")
def replay(
    scenarios_dir: Path | None,
    reports_dir: Path | None,
    verbose: bool,
) -> None:
    """Run offline replay scenarios and report pass/fail for each."""
    from autogpt.config import ConfigBuilder
    from autogpt.core.self_improvement.replay import ScenarioLoader
    from autogpt.core.self_improvement.replay_validator import ReplayValidator

    config = ConfigBuilder.build_config_from_env()
    learning_cfg = config.learning

    scenarios_path = scenarios_dir or Path(learning_cfg.replay_scenarios_dir)
    reports_path = reports_dir or Path(learning_cfg.replay_reports_dir)

    loader = ScenarioLoader(scenarios_path)
    scenarios = loader.load()
    if not scenarios:
        click.echo(f"No replay scenarios found in {scenarios_path}")
        raise SystemExit(0)

    validator = ReplayValidator(reports_path, logger=logging.getLogger("ReplayValidator"))

    def _execute(scenario):
        # Placeholder executor: succeeds when all records in scenario have success status
        return all(
            record.result_status and record.result_status.lower() == "success"
            for record in scenario.records
        )

    results = validator.evaluate_scenarios(scenarios, _execute)
    scenario_map = {scenario.name: scenario for scenario in scenarios}
    overall_pass = True
    for name, passed in results.items():
        overall_pass &= passed
        status = "PASS" if passed else "FAIL"
        click.echo(f"{name}: {status}")
        if verbose:
            scenario = scenario_map.get(name)
            size = len(scenario.records) if scenario else 0
            click.echo(f"  Records: {size}")
    if overall_pass:
        click.echo("All replay scenarios passed")
    else:
        click.echo("Some replay scenarios failed", err=True)
        raise SystemExit(1)



if __name__ == "__main__":
    cli()
