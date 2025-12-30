"""Evolution package containing specialized agent implementations."""

from abc import ABC, abstractmethod

from .replay_buffer import ReplayBuffer
from .evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
    GeneticAlgorithm as EvolutionGeneticAlgorithm,
)
from .self_evolving_cognition import SelfEvolvingCognition
from .self_evolving_ai_architecture import SelfEvolvingAIArchitecture
from .evolution_engine import (
    EvolutionEngine,
    SpecialistModule,
    SpecialistModuleRegistry,
    TaskContext,
)
from .cognitive_benchmark import (
    CognitiveBenchmarkResult,
    aggregate_benchmark_score,
    summarise_benchmarks,
)
from .nas import NASMutationSpace, NASParameter
from .meta_nas import MetaNASController, UCBBanditSelector, NASCandidate
from .mutation_operators import (
    MutationContext,
    MutationOperator,
    MutationOperatorLibrary,
    default_operator_library,
)
from .multiobjective import MultiObjectiveConfig, adjust_performance, summarize_metric_events
from .evolution_recorder import EvolutionKnowledgeRecorder
from .adapter import EvolutionModule
from .dynamic_architecture import DynamicArchitectureExpander
from .structural_genome import StructuralGenome, ModuleGene, EdgeGene
from .structural_evolution import StructuralEvolutionManager, StructuralProposal
from .neuroevolution_backend import (
    CognitiveConnectionGene,
    CognitiveNetworkGenome,
    CognitiveNodeGene,
    NeuroevolutionBackend,
)
from .structural_encoding import encode_structure
from .strategy_adjuster import StrategyAdjuster, StrategyAction
from .agent_self_improvement import (
    AgentSelfImprovementController,
    SelfImprovementUpdate,
    StrategyGenome,
)
from .safety import (
    ArchitectureApprovalQueue,
    EvolutionSafetyConfig,
    PendingArchitectureUpdate,
    PytestSandboxRunner,
    SafetyDecision,
    architecture_delta_l1,
    evaluate_candidate,
)
from .strategy import (
    ExplorationStrategy,
    SimulatedAnnealingStrategy,
    InnovationProtectionStrategy,
)

try:  # optional dependencies
    from .ppo import PPO, PPOConfig
    from .a3c import A3C, A3CConfig
    from .sac import SAC, SACConfig
except Exception:  # pragma: no cover - algorithms require torch
    PPO = PPOConfig = A3C = A3CConfig = SAC = SACConfig = None  # type: ignore

try:  # Attempt to leverage AutoGPT's agent base when available
    from third_party.autogpt.autogpt.agents.base import BaseAgent as AutoGPTBaseAgent
except Exception:  # pragma: no cover - fallback when dependencies missing
    AutoGPTBaseAgent = None  # type: ignore


if AutoGPTBaseAgent is None:
    class Agent(ABC):  # type: ignore[too-many-ancestors]
        """Abstract base agent for the evolution package.

        Subclasses should implement :meth:`perform`, which executes the agent's
        primary behaviour.
        """

        @abstractmethod
        def perform(self, *args, **kwargs):
            """Execute the agent's primary behaviour."""
            raise NotImplementedError


else:
    class Agent(AutoGPTBaseAgent, ABC):
        """Abstract base agent for the evolution package.

        Subclasses should implement :meth:`perform`, which executes the agent's
        primary behaviour. When AutoGPT's BaseAgent is available the class
        inherits from it, enabling integration with the broader AutoGPT
        ecosystem.
        """

        @abstractmethod
        def perform(self, *args, **kwargs):
            """Execute the agent's primary behaviour."""
            raise NotImplementedError


__all__ = [
    "Agent",
    "ReplayBuffer",
    "EvolvingCognitiveArchitecture",
    "EvolutionGeneticAlgorithm",
    "SelfEvolvingCognition",
    "SelfEvolvingAIArchitecture",
    "EvolutionEngine",
    "SpecialistModule",
    "SpecialistModuleRegistry",
    "TaskContext",
    "CognitiveBenchmarkResult",
    "aggregate_benchmark_score",
    "summarise_benchmarks",
    "NASParameter",
    "NASMutationSpace",
    "MetaNASController",
    "UCBBanditSelector",
    "NASCandidate",
    "MutationContext",
    "MutationOperator",
    "MutationOperatorLibrary",
    "default_operator_library",
    "MultiObjectiveConfig",
    "adjust_performance",
    "summarize_metric_events",
    "EvolutionKnowledgeRecorder",
    "DynamicArchitectureExpander",
    "StructuralGenome",
    "ModuleGene",
    "EdgeGene",
    "CognitiveNetworkGenome",
    "CognitiveConnectionGene",
    "CognitiveNodeGene",
    "NeuroevolutionBackend",
    "StructuralEvolutionManager",
    "StructuralProposal",
    "encode_structure",
    "StrategyAdjuster",
    "StrategyAction",
    "AgentSelfImprovementController",
    "SelfImprovementUpdate",
    "StrategyGenome",
    "EvolutionSafetyConfig",
    "SafetyDecision",
    "PendingArchitectureUpdate",
    "ArchitectureApprovalQueue",
    "PytestSandboxRunner",
    "architecture_delta_l1",
    "evaluate_candidate",
    "ExplorationStrategy",
    "SimulatedAnnealingStrategy",
    "InnovationProtectionStrategy",
    "PPO",
    "PPOConfig",
    "A3C",
    "A3CConfig",
    "SAC",
    "SACConfig",
    "EvolutionModule",
]
