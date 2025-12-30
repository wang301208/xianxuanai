"""Long-term memory persistence utilities."""

from .embedders import HashingEmbedder, TransformerEmbedder, TextEmbedder
from .task_memory import ExperiencePayload, TaskMemoryManager
from .lifecycle import MemoryLifecycleManager
from .experience_bridge import ExperienceMemoryBridge, curiosity_weighted_summarizer
from .vector_store import VectorMemoryStore, VectorRecord
from .backends import (
    VectorANNBackend,
    VectorArchiveBackend,
    InMemoryANNBackend,
    FileArchiveBackend,
    S3ArchiveBackend,
    build_ann_backend_from_env,
    build_archive_backend_from_env,
)
from .maintenance import MemoryDecayPolicy, MemoryMaintenanceDaemon
from .vector_shards import VectorShardCoordinator, create_ray_vector_coordinator

__all__ = [
    "HashingEmbedder",
    "TransformerEmbedder",
    "TextEmbedder",
    "VectorANNBackend",
    "VectorArchiveBackend",
    "InMemoryANNBackend",
    "FileArchiveBackend",
    "S3ArchiveBackend",
    "build_ann_backend_from_env",
    "build_archive_backend_from_env",
    "TaskMemoryManager",
    "ExperiencePayload",
    "ExperienceMemoryBridge",
    "curiosity_weighted_summarizer",
    "VectorMemoryStore",
    "VectorRecord",
    "MemoryDecayPolicy",
    "MemoryMaintenanceDaemon",
    "MemoryLifecycleManager",
    "VectorShardCoordinator",
    "create_ray_vector_coordinator",
]
