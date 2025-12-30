from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm.

    Values can be overridden through environment variables prefixed with
    ``GA_`` or by passing keyword arguments to :meth:`update`.
    """

    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    max_generations: int = 100
    early_stopping_rounds: int = 10
    target_fitness: float | None = None
    n_workers: int = os.cpu_count() or 1
    gene_length: int = 10
    cache_path: Path = Path("evolution/ga_cache.pkl")

    @classmethod
    def from_env(cls) -> "GAConfig":
        """Create a configuration from ``GA_*`` environment variables."""

        kwargs: dict[str, object] = {}
        for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            env_var = f"GA_{field_name.upper()}"
            if env_var in os.environ:
                value = os.environ[env_var]
                field_type = field_def.type
                try:
                    if field_type in (int, float):
                        kwargs[field_name] = field_type(value)
                    elif field_type is Path:
                        kwargs[field_name] = Path(value)
                    else:
                        kwargs[field_name] = value
                except Exception:
                    continue
        return cls(**kwargs)

    def update(self, **kwargs: object) -> "GAConfig":
        """Update configuration values and return ``self`` for chaining."""

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
