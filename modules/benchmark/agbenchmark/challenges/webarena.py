import difflib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import ClassVar, Iterator, Iterable, Literal

import pytest
import requests
from agent_protocol_client import AgentApi, Step
from pydantic import BaseModel, validator, ValidationError

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, EvalResult

from bs4 import BeautifulSoup

from .base import BaseChallenge, ChallengeInfo

logger = logging.getLogger(__name__)


EvalType = Literal["string_match", "url_match", "program_html"]
WebArenaSite = Literal[
    "gitlab", "map", "reddit", "shopping", "shopping_admin", "wikipedia"
]
ReferenceAnswerType = Literal["exact_match", "fuzzy_match", "must_include"]


class WebArenaSiteInfo(BaseModel):
    base_url: str
    available: bool = True
    additional_info: str = ""
    unavailable_reason: str = ""


_git_user, _git_password = os.getenv("WEBARENA_GIT_CREDENTIALS", ":").split(":")

site_info_map: dict[WebArenaSite, WebArenaSiteInfo] = {
    "gitlab": WebArenaSiteInfo(
        base_url="http://git.junglegym.ai",
        available=bool(_git_user and _git_password),
        additional_info=(
            f"To log in to {{url}}, use the username '{_git_user}' "
            f"and password '{_git_password}'."
        ),
        unavailable_reason=(
            "WEBARENA_GIT_CREDENTIALS not set (correctly): "
            f"'{os.getenv('WEBARENA_GIT_CREDENTIALS', '')}', "
            "should be USERNAME:PASSWORD."
        ),
    ),
    "map": WebArenaSiteInfo(
        base_url="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/"
    ),
    "reddit": WebArenaSiteInfo(base_url="http://forum.junglegym.ai"),
    "shopping": WebArenaSiteInfo(base_url="http://shop.junglegym.ai"),
    "shopping_admin": WebArenaSiteInfo(
        base_url="http://cms.junglegym.ai/admin",
        additional_info=(
            "To log in to {url}, use the username 'admin' and password 'admin1234'."
        ),
    ),
    "wikipedia": WebArenaSiteInfo(base_url="http://wiki.junglegym.ai"),
}

SITE_CATEGORY_OVERRIDES: dict[WebArenaSite, list[Category]] = {
    "gitlab": [Category.CODING, Category.DATA],
    "map": [Category.DATA, Category.GENERALIST],
    "reddit": [Category.WEB, Category.SCRAPE_SYNTHESIZE],
    "shopping": [Category.WEB, Category.DATA, Category.SCRAPE_SYNTHESIZE],
    "shopping_admin": [Category.WEB, Category.DATA, Category.SCRAPE_SYNTHESIZE],
    "wikipedia": [Category.WEB, Category.DATA],
}


def _unique_ordered_categories(categories: Iterable[Category]) -> list[Category]:
    seen: set[Category] = set()
    ordered: list[Category] = []
    for category in categories:
        if category not in seen:
            seen.add(category)
            ordered.append(category)
    return ordered


def _categories_for_sites(sites: list[WebArenaSite]) -> list[Category]:
    categories: list[Category] = []
    for site in sites:
        categories.extend(SITE_CATEGORY_OVERRIDES.get(site, [Category.WEB]))
    if not categories:
        categories.append(Category.GENERALIST)
    return _unique_ordered_categories(categories)


def get_site_info(site: WebArenaSite) -> WebArenaSiteInfo:
    if site not in site_info_map:
        raise ValueError(f"JungleGym site '{site}' unknown, cannot resolve URL")
    return site_info_map[site]


def get_site_url(site: WebArenaSite) -> str:
    return get_site_info(site).base_url


def resolve_uri(uri: str) -> str:
    """
    Resolves URIs with mock hosts, like `__WIKI__/wiki/Octopus`, with the corresponding
    JungleGym site mirror host.
    """
    segments = uri.split("__")
    if len(segments) > 2 and (site := segments[1]).lower() in site_info_map:
        return uri.replace(f"__{site}__", get_site_url(site.lower()))  # type: ignore
    return uri


class Eval(ABC):
    @abstractmethod
    def evaluate(self, string: str) -> bool:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...


class StringEval(BaseModel, Eval):
    type: ReferenceAnswerType


class ExactStringMatchEval(StringEval):
    type: Literal["exact_match"] = "exact_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must be '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return string == self.reference_answer


class FuzzyStringMatchEval(StringEval):
    type: Literal["fuzzy_match"] = "fuzzy_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must contain something like '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        candidate = string or ""
        expected = self.reference_answer or ""
        if not candidate or not expected:
            return False
        candidate_lower = candidate.lower()
        expected_lower = expected.lower()
        ratio = difflib.SequenceMatcher(None, expected_lower, candidate_lower).ratio()
        if ratio >= 0.65:
            return True
        for word in expected_lower.split():
            if word and word in candidate_lower:
                return True
        return False


class MustIncludeStringEval(StringEval):
    type: Literal["must_include"] = "must_include"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must include '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return self.reference_answer.lower() in string.lower()


class UrlMatchEval(BaseModel, Eval):
    url: str
    """Example: `"__WIKI__/wiki/Octopus"`"""

    @property
    def description(self) -> str:
        return f"Agent must navigate to '{self.url}'"

    def evaluate(self, url: str) -> bool:
        return url == resolve_uri(self.url)


class ProgramHtmlEval(BaseModel):
    url: str
    locator: str
    """JavaScript code that returns the value to check"""
    required_contents: str

    @property
    def description(self) -> str:
        return (
            f"On the webpage {self.url}, "
            f"`{self.locator}` should contain '{self.required_contents}'"
        )

    def evaluate(self, selenium_instance) -> bool:
        result = selenium_instance.execute_script(
            self.locator or "return document.body.innerHTML;"
        )
        return self.required_contents in result


_Eval = StringEval | UrlMatchEval | ProgramHtmlEval


class WebArenaChallengeSpec(BaseModel):
    task_id: int
    sites: list[WebArenaSite]
    """The sites needed to complete the task"""
    start_url: str
    """The full URL at which to start"""
    start_url_junglegym: str
    """The JungleGym site (base URL) at which to start"""
    require_login: bool
    require_reset: bool
    storage_state: str | None

    intent: str
    intent_template: str
    intent_template_id: int
    instantiation_dict: dict[str, str | list[str]]

    available: bool = True
    unavailable_reason: str = ""

    class EvalSet(BaseModel):
        class StringMatchEvalSet(BaseModel):
            exact_match: str | None
            fuzzy_match: list[str] | None
            must_include: list[str] | None

        reference_answers: StringMatchEvalSet | None
        """For string_match eval, a set of criteria to judge the final answer"""
        reference_answer_raw_annotation: str | None
        string_note: str | None
        annotation_note: str | None

        reference_url: str | None
        """For url_match eval, the last URL that should be visited"""
        url_note: str | None

        program_html: list[ProgramHtmlEval]
        """For program_html eval, a list of criteria to judge the site state by"""

        eval_types: list[EvalType]

        @validator("eval_types")
        def check_eval_parameters(cls, v: list[EvalType], values):
            if "string_match" in v and not values.get("reference_answers"):
                raise ValueError("'string_match' eval_type requires reference_answers")
            if "url_match" in v and not values.get("reference_url"):
                raise ValueError("'url_match' eval_type requires reference_url")
            if "program_html" in v and not values.get("program_html"):
                raise ValueError(
                    "'program_html' eval_type requires at least one program_html eval"
                )
            return v

        @property
        def evaluators(self) -> list[_Eval]:
            evaluators: list[_Eval] = []
            if self.reference_answers:
                if self.reference_answers.exact_match:
                    evaluators.append(
                        ExactStringMatchEval(
                            reference_answer=self.reference_answers.exact_match
                        )
                    )
                if self.reference_answers.fuzzy_match:
                    evaluators.extend(
                        FuzzyStringMatchEval(reference_answer=a)
                        for a in self.reference_answers.fuzzy_match
                    )
                if self.reference_answers.must_include:
                    evaluators.extend(
                        MustIncludeStringEval(reference_answer=a)
                        for a in self.reference_answers.must_include
                    )
            if self.reference_url:
                evaluators.append(UrlMatchEval(url=self.reference_url))
            evaluators.extend(self.program_html)
            return evaluators

    eval: EvalSet
    """Evaluation criteria by which to judge the agent's performance"""

    @property
    def assignment_for_agent(self):
        sites = [get_site_info(s) for s in self.sites]
        nav_constraint = (
            "You are ONLY allowed to access URLs in "
            f"{' and '.join(s.base_url for s in sites)}.\n\n"
            + "\n".join(
                s.additional_info.format(url=s.base_url)
                for s in sites if s.additional_info
            )
        ).strip()

        return (
            f"First of all, go to {self.start_url}. "
            f"{self.intent.rstrip('.')}.\n"
            f"{nav_constraint}"
        )


class WebArenaChallenge(BaseChallenge):
    _spec: ClassVar[WebArenaChallengeSpec]

    SOURCE_URI_PREFIX = "__JUNGLEGYM__/webarena/tasks/"
    SOURCE_URI_TEMPLATE = f"{SOURCE_URI_PREFIX}{{task_id}}"
    URL_CANDIDATE_KEYS: ClassVar[tuple[str, ...]] = (
        "browser_url",
        "current_url",
        "page_url",
        "target_url",
        "url",
    )
    HTML_SNAPSHOT_KEYS: ClassVar[tuple[str, ...]] = (
        "browser_dom",
        "browser_html",
        "page_html",
        "page_source",
        "document_html",
        "html_snapshot",
    )
    URL_PATTERN: ClassVar[re.Pattern] = re.compile(r"https?://[^\s'\"<>]+")

    @classmethod
    def from_source_uri(cls, source_uri: str) -> type["WebArenaChallenge"]:
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for WebArenaChallenge: {source_uri}")

        source_url = source_uri.replace(
            cls.SOURCE_URI_PREFIX,
            "https://api.junglegym.ai/get_webarena_by_task_id?task_id=",
        )
        results = requests.get(source_url).json()["data"]
        if not results:
            raise ValueError(f"Could not fetch challenge {source_uri}")
        return cls.from_challenge_spec(WebArenaChallengeSpec.parse_obj(results[0]))

    @classmethod
    def from_challenge_spec(
        cls, spec: WebArenaChallengeSpec
    ) -> type["WebArenaChallenge"]:
        challenge_info = ChallengeInfo(
            eval_id=f"junglegym-webarena-{spec.task_id}",
            name=f"WebArenaTask_{spec.task_id}",
            task=spec.assignment_for_agent,
            category=_categories_for_sites(spec.sites),
            reference_answer=spec.eval.reference_answer_raw_annotation,
            source_uri=cls.SOURCE_URI_TEMPLATE.format(task_id=spec.task_id),
            available=spec.available,
            unavailable_reason=spec.unavailable_reason,
        )
        return type(
            f"Test{challenge_info.name}",
            (WebArenaChallenge,),
            {
                "info": challenge_info,
                "_spec": spec,
            },
        )

    @classmethod
    def evaluate_answer(cls, answer: str) -> list[tuple[_Eval, EvalResult]]:
        if not answer:
            return []
        results: list[tuple[_Eval, EvalResult]] = []
        for evaluator in cls._spec.eval.evaluators:
            if isinstance(evaluator, StringEval):  # string_match
                passed = evaluator.evaluate(answer)
                results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=answer,
                            result_source="step_output",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
        return results

    @classmethod
    def evaluate_step_result(
        cls, step: Step, *, mock: bool = False
    ) -> list[tuple[_Eval, EvalResult]]:
        if mock:
            step.output = cls.info.reference_answer
        string_output = step.output or ""
        url_candidates = cls._collect_urls_from_step(step)
        html_proxy = cls._create_html_proxy(step)
        eval_results: list[tuple[_Eval, EvalResult]] = []
        for evaluator in cls._spec.eval.evaluators:
            if isinstance(evaluator, StringEval):
                if not string_output:
                    continue
                passed = evaluator.evaluate(string_output)
                eval_results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=string_output,
                            result_source="step_output",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
            elif isinstance(evaluator, UrlMatchEval):
                passed = cls._match_url(evaluator, url_candidates)
                eval_results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=", ".join(url_candidates) or string_output,
                            result_source="url_candidates",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
            elif isinstance(evaluator, ProgramHtmlEval):
                if html_proxy is None:
                    eval_results.append(
                        (
                            evaluator,
                            EvalResult(
                                result="",
                                result_source="html_snapshot",
                                score=0.0,
                                passed=False,
                            ),
                        )
                    )
                    continue
                passed = evaluator.evaluate(html_proxy)
                eval_results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=html_proxy.html,
                            result_source="html_snapshot",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
        return eval_results

    @classmethod
    def _collect_urls_from_step(cls, step: Step) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        additional = step.additional_output
        if isinstance(additional, dict):
            for key in cls.URL_CANDIDATE_KEYS:
                value = additional.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed and trimmed not in seen:
                        seen.add(trimmed)
                        candidates.append(trimmed)
        if step.output:
            for match in cls.URL_PATTERN.finditer(step.output):
                url = match.group(0)
                if url not in seen:
                    seen.add(url)
                    candidates.append(url)
            trimmed_output = step.output.strip()
            if trimmed_output and trimmed_output not in seen:
                seen.add(trimmed_output)
                candidates.append(trimmed_output)
        return candidates

    @classmethod
    def _extract_html_snapshot(cls, step: Step) -> str | None:
        additional = step.additional_output
        if isinstance(additional, dict):
            for key in cls.HTML_SNAPSHOT_KEYS:
                value = additional.get(key)
                if isinstance(value, str):
                    candidate = value.strip()
                    if candidate:
                        return candidate
        output = step.output or ""
        if "<html" in output.lower() or "<body" in output.lower():
            return output
        return None

    @classmethod
    def _create_html_proxy(cls, step: Step) -> "_HtmlSnapshot" | None:
        html = cls._extract_html_snapshot(step)
        if html:
            return _HtmlSnapshot(html)
        return None

    @staticmethod
    def _match_url(evaluator: UrlMatchEval, candidates: Iterable[str]) -> bool:
        target = resolve_uri(evaluator.url)
        for candidate in candidates:
            if candidate == target or target in candidate:
                return True
        return False

    @classmethod
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    ) -> list[EvalResult]:
        steps: list[Step] = (await agent.list_agent_task_steps(task_id)).steps

        eval_results_per_step = [cls.evaluate_step_result(step) for step in steps]
        # Get the column aggregate (highest scored EvalResult for each Eval)
        # from the matrix of EvalResults per step.
        return [
            max(step_results_for_eval, key=lambda r: r[1].score)[1]
            for step_results_for_eval in zip(*eval_results_per_step)
        ]

    @pytest.mark.asyncio
    async def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
    ) -> None:
        if not self._spec.available:
            pytest.skip(self._spec.unavailable_reason)

        # if os.environ.get("HELICONE_API_KEY"):
        #     from helicone.lock import HeliconeLockManager

        #     HeliconeLockManager.write_custom_property("challenge", self.info.name)

        timeout = 120
        if request.config.getoption("--nc"):
            timeout = 100000
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)

        n_steps = 0
        timed_out = None
        agent_task_cost = None
        steps: list[Step] = []
        eval_results_per_step: list[list[tuple[_Eval, EvalResult]]] = []
        try:
            async for step in self.run_challenge(
                config, timeout, mock=request.config.getoption("--mock")
            ):
                if not step.output:
                    logger.warn(f"Step has no output: {step}")
                    continue

                n_steps += 1
                steps.append(step)
                if step.additional_output:
                    agent_task_cost = step.additional_output.get(
                        "task_total_cost",
                        step.additional_output.get("task_cumulative_cost"),
                    )

                step_eval_results = self.evaluate_step_result(
                    step, mock=request.config.getoption("--mock")
                )
                logger.debug(f"Intermediary results: {step_eval_results}")
                eval_results_per_step.append(step_eval_results)
                if step.is_last:
                    request.node.user_properties.append(
                        (
                            "answers",
                            step.output
                            if request.config.getoption("--keep-answers")
                            else None,
                        )
                    )
            timed_out = False
        except TimeoutError:
            timed_out = True
        request.node.user_properties.append(("steps", steps))
        request.node.user_properties.append(("n_steps", n_steps))
        request.node.user_properties.append(("timed_out", timed_out))
        request.node.user_properties.append(("agent_task_cost", agent_task_cost))

        # Get the column aggregate (highest score for each Eval)
        # from the matrix of EvalResults per step.
        evals_results = [
            max(step_results_for_eval, key=lambda r: r[1].score)
            for step_results_for_eval in zip(*eval_results_per_step)
        ]

        if not evals_results:
            if timed_out:
                raise TimeoutError("Timed out, no results to evaluate")
            else:
                raise ValueError("No results to evaluate")

        request.node.user_properties.append(
            ("scores", [r[1].score for r in evals_results])
        )

        # FIXME: arbitrary threshold
        assert all(r[1].score > 0.9 for r in evals_results), (
            "Scores insufficient:\n\n"
            if not timed_out
            else "Timed out; scores insufficient:\n\n"
        ) + "\n".join(f"{repr(r[0])}\n  -> {repr(r[1])}" for r in evals_results)


class _HtmlSnapshot:
    """Lightweight DOM proxy that satisfies the limited scripts used in tests."""

    QUERY_SELECTOR_PATTERN = re.compile(
        r"document\.querySelector(All)?\(\s*(['\"])(.+?)\2\s*\)\.([a-zA-Z0-9_]+)"
    )

    def __init__(self, html: str) -> None:
        self.html = html or ""
        self._soup = BeautifulSoup(self.html, "html.parser")

    def execute_script(self, script: str) -> str:
        if not script:
            return self.html
        trimmed = script.strip()
        if trimmed.endswith(";"):
            trimmed = trimmed[:-1].strip()
        if trimmed.startswith("return "):
            trimmed = trimmed[len("return ") :].strip()
        if not trimmed:
            return self.html

        lower_script = trimmed.lower()
        if lower_script in (
            "document.body.innerhtml",
            "document.documentelement.innerhtml",
        ):
            return self.html
        if lower_script in (
            "document.body.textcontent",
            "document.documentelement.textcontent",
        ):
            return self._soup.get_text()

        match = self.QUERY_SELECTOR_PATTERN.search(trimmed)
        if match:
            selector = match.group(3)
            prop = match.group(4)
            element = self._soup.select_one(selector)
            if not element:
                return ""
            prop_lower = prop.lower()
            if prop_lower in {"textcontent", "innertext"}:
                return element.get_text()
            if prop_lower == "innerhtml":
                return "".join(str(child) for child in element.contents)
            attribute_value = element.get(prop)
            if attribute_value is not None:
                return str(attribute_value)
            return element.get_text()

        if "textcontent" in lower_script:
            return self._soup.get_text()
        if "innerhtml" in lower_script:
            return self.html
        return self.html


def load_webarena_challenges(
    skip_unavailable: bool = True
) -> Iterator[type[WebArenaChallenge]]:
    logger.info("Loading WebArena challenges...")

    for site, info in site_info_map.items():
        if not info.available and skip_unavailable:
            logger.warning(
                f"JungleGym site '{site}' is not available: {info.unavailable_reason} "
                "Skipping all challenges which use this site."
            )

    # response = requests.get("https://api.junglegym.ai/get_full_webarena_dataset")
    # challenge_dicts = response.json()["data"]

    # Until the full WebArena challenge set is supported, use a hand-picked selection
    import json
    from pathlib import Path

    challenge_dicts = json.loads(
        (Path(__file__).parent / "webarena_selection.json").read_bytes()
    )

    logger.debug(
        "Fetched WebArena dataset. "
        f"Constructing {len(challenge_dicts)} WebArenaChallenges..."
    )
    loaded = 0
    failed = 0
    skipped = 0
    for entry in challenge_dicts:
        try:
            challenge_spec = WebArenaChallengeSpec.parse_obj(entry)
        except ValidationError as e:
            failed += 1
            logger.warning(f"Error validating WebArena challenge entry: {entry}")
            logger.warning(f"Error details: {e}")
            continue

        # Check all required sites for availability
        for site in challenge_spec.sites:
            site_info = site_info_map.get(site)
            if site_info is None:
                challenge_spec.available = False
                challenge_spec.unavailable_reason = (
                    f"WebArena task {challenge_spec.task_id} requires unknown site "
                    f"'{site}'"
                )
            elif not site_info.available:
                challenge_spec.available = False
                challenge_spec.unavailable_reason = (
                    f"WebArena task {challenge_spec.task_id} requires unavailable "
                    f"site '{site}'"
                )

        if not challenge_spec.available and skip_unavailable:
            logger.debug(f"{challenge_spec.unavailable_reason}; skipping...")
            skipped += 1
            continue

        yield WebArenaChallenge.from_challenge_spec(challenge_spec)
        loaded += 1

    logger.info(
        "Loading WebArena challenges complete: "
        f"loaded {loaded}, skipped {skipped}."
        + (f" {failed} challenges failed to load." if failed else "")
    )
