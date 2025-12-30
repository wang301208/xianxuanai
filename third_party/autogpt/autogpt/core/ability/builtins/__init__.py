from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.builtins.file_operations import ReadFile, WriteFile
from autogpt.core.ability.builtins.generate_tests import GenerateTests
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel
from autogpt.core.ability.builtins.run_tests import RunTests
from autogpt.core.ability.builtins.evaluate_metrics import EvaluateMetrics
from autogpt.core.ability.builtins.lint_code import LintCode
from autogpt.core.ability.builtins.knowledge_query import KnowledgeQuery
from autogpt.core.ability.builtins.symbolic_reason import SymbolicReason
from autogpt.core.ability.builtins.causal_reason import CausalReason
from autogpt.core.ability.builtins.commonsense_validate import CommonsenseValidate
from autogpt.core.ability.builtins.web_search import WebSearch
from autogpt.core.ability.builtins.web_scrape import WebScrape
from autogpt.core.ability.builtins.web_search_and_scrape import WebSearchAndScrape
from autogpt.core.ability.builtins.github_code_search import GitHubCodeSearch
from autogpt.core.ability.builtins.documentation_tool import DocumentationTool

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
    CreateNewAbility.name(): CreateNewAbility,
    ReadFile.name(): ReadFile,
    WriteFile.name(): WriteFile,
    RunTests.name(): RunTests,
    GenerateTests.name(): GenerateTests,
    EvaluateMetrics.name(): EvaluateMetrics,
    LintCode.name(): LintCode,
    KnowledgeQuery.name(): KnowledgeQuery,
    WebSearch.name(): WebSearch,
    WebScrape.name(): WebScrape,
    WebSearchAndScrape.name(): WebSearchAndScrape,
    GitHubCodeSearch.name(): GitHubCodeSearch,
    DocumentationTool.name(): DocumentationTool,
    SymbolicReason.name(): SymbolicReason,
    CausalReason.name(): CausalReason,
    CommonsenseValidate.name(): CommonsenseValidate,
}

__all__ = [
    "BUILTIN_ABILITIES",
    "CreateNewAbility",
    "QueryLanguageModel",
    "ReadFile",
    "WriteFile",
    "RunTests",
    "GenerateTests",
    "EvaluateMetrics",
    "LintCode",
    "KnowledgeQuery",
    "WebSearch",
    "WebScrape",
    "WebSearchAndScrape",
    "GitHubCodeSearch",
    "DocumentationTool",
    "SymbolicReason",
    "CausalReason",
    "CommonsenseValidate",
]
