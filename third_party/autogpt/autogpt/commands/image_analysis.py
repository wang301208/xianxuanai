"""Image analysis command leveraging CLIP feature extraction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - runtime environments without pillow
    Image = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from PIL import Image as PILImage
else:  # Fallback type alias when pillow is unavailable
    PILImage = Any

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.commands.decorators import sanitize_path_arg
from autogpt.core.utils.json_schema import JSONSchema

COMMAND_CATEGORY = "image_analysis"
COMMAND_CATEGORY_TITLE = "Image Analysis"

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - runtime environments without numpy
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import CLIPFeatureExtractor  # type: ignore
except Exception:  # pragma: no cover - runtime fallback when transformers missing
    CLIPFeatureExtractor = None  # type: ignore


@dataclass
class _ClipHandles:
    extractor: Any | None
    used_model: Optional[str]


def _load_feature_extractor(model_name: Optional[str] = None) -> _ClipHandles:
    """Load a ``CLIPFeatureExtractor`` instance if the dependency is available."""

    if CLIPFeatureExtractor is None:
        return _ClipHandles(extractor=None, used_model=None)

    names_to_try = []
    if model_name:
        names_to_try.append(model_name)
    names_to_try.append("openai/clip-vit-base-patch32")

    for name in names_to_try:
        try:
            extractor = CLIPFeatureExtractor.from_pretrained(name)  # type: ignore[attr-defined]
            return _ClipHandles(extractor=extractor, used_model=name)
        except Exception:  # pragma: no cover - runtime environments without weights
            continue

    try:
        extractor = CLIPFeatureExtractor()  # type: ignore[call-arg]
        return _ClipHandles(extractor=extractor, used_model=None)
    except Exception:  # pragma: no cover - final fallback when constructor fails
        logger.warning("Failed to initialise CLIPFeatureExtractor; falling back to pixel stats")
        return _ClipHandles(extractor=None, used_model=None)


def _flatten_numeric(value: Any) -> list[float]:
    if np is not None:
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            return arr.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    try:
        return [float(value)]
    except Exception:
        return []


def _downsample_vector(data: Any, *, max_dim: int = 64) -> list[float]:
    flat = _flatten_numeric(data)
    if not flat:
        return []
    if len(flat) <= max_dim or max_dim <= 1:
        return flat[:max_dim]
    step = (len(flat) - 1) / (max_dim - 1)
    return [flat[int(round(i * step))] for i in range(max_dim)]


def _mean_rgb_from_pixels(pixels: Iterable[tuple[int, int, int]]) -> tuple[float, float, float]:
    total = 0
    accum = [0.0, 0.0, 0.0]
    for r, g, b in pixels:
        accum[0] += float(r)
        accum[1] += float(g)
        accum[2] += float(b)
        total += 1
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (accum[0] / total, accum[1] / total, accum[2] / total)


def _clip_features(
    extractor: Any | None,
    *,
    image: PILImage,
    description: str,
    mean_rgb: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    if extractor is None:
        image_stats = [channel / 255.0 for channel in mean_rgb]
        text_stats = [float(len(description)), float(description.count(" ") + 1)]
        return image_stats, text_stats

    image_output = extractor(images=image, return_tensors="np")
    image_values: list[float] | None = None
    if isinstance(image_output, dict):
        for value in image_output.values():
            image_values = _downsample_vector(value)
            if image_values:
                break
    if not image_values:
        image_values = _downsample_vector([channel / 255.0 for channel in mean_rgb])

    text_features: list[float] | None = None
    try:
        text_output = extractor(text=[description], return_tensors="np")
    except Exception:
        text_output = None
    if isinstance(text_output, dict):
        for value in text_output.values():
            text_features = _downsample_vector(value)
            if text_features:
                break
    if not text_features:
        text_features = [float(len(description)), float(description.count(" ") + 1)]

    return image_values, text_features


def _describe_image(
    image: PILImage,
    *,
    style: str,
    mean_rgb: tuple[float, float, float],
) -> str:
    width, height = image.size
    mode = image.mode
    fmt = image.format or "unknown format"
    mean_color = tuple(int(round(channel)) for channel in mean_rgb)
    dominant_color = f"RGB{mean_color}"

    if style == "detailed":
        description = (
            f"The image measures {width}x{height} pixels in {mode} mode (stored as {fmt}). "
            f"Average colour intensity approximates {dominant_color}."
        )
    elif style == "technical":
        description = (
            f"Resolution={width}x{height}, mode={mode}, format={fmt}. "
            f"Mean RGB intensity: {dominant_color}."
        )
    else:
        description = f"Image of size {width}x{height} with average colour {dominant_color}."

    return description


def _write_summary(
    agent: Agent,
    *,
    image_path: Path,
    description: str,
    style: str,
    image_features: list[float],
    text_features: list[float] | None,
    model_name: Optional[str],
) -> Path:
    analysis_dir = agent.workspace.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / f"{image_path.stem}_analysis.json"
    summary = {
        "image": image_path.name,
        "style": style,
        "description": description,
        "model": model_name,
        "image_features": list(image_features),
        "text_features": list(text_features) if text_features is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


@sanitize_path_arg("image_path", make_relative=True)
@command(
    "analyze_image",
    "Analyse an image and store descriptive features",
    {
        "image_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Path to the image relative to the agent workspace",
            required=True,
        ),
        "description_style": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Optional narrative style for the generated description",
            enum=["concise", "detailed", "technical"],
            required=False,
        ),
    },
)
def analyze_image(
    image_path: Path,
    agent: Agent,
    description_style: str | None = None,
    *,
    model_name: str | None = None,
) -> Dict[str, Any]:
    """Process an image using CLIP feature extraction and log the findings."""

    style = (description_style or "concise").lower()
    if style not in {"concise", "detailed", "technical"}:
        logger.warning("Unknown description style '%s', falling back to concise", style)
        style = "concise"

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if Image is None:
        raise RuntimeError("Pillow is required to run analyze_image")

    with Image.open(image_path) as pil_image:
        pil_image.load()
        rgb_pixels = list(pil_image.convert("RGB").getdata())
        mean_rgb = _mean_rgb_from_pixels(rgb_pixels)
        description = _describe_image(pil_image, style=style, mean_rgb=mean_rgb)

        clip_handles = _load_feature_extractor(model_name)
        image_features, text_features = _clip_features(
            clip_handles.extractor,
            image=pil_image,
            description=description,
            mean_rgb=mean_rgb,
        )
        image_array = rgb_pixels

    summary_path = _write_summary(
        agent,
        image_path=image_path,
        description=description,
        style=style,
        image_features=image_features,
        text_features=text_features,
        model_name=clip_handles.used_model,
    )

    record_visual = getattr(agent, "record_visual_observation", None)
    world_model_updated = False
    text_payload: Dict[str, Any] = {
        "description": description,
        "style": style,
        "features": list(text_features) if text_features is not None else None,
    }
    if callable(record_visual):
        try:
            record_visual(features=image_features, text=text_payload)
            world_model_updated = True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to record visual observation: %s", exc)
    else:
        world_model = getattr(agent, "world_model", None)
        agent_id = getattr(getattr(agent, "settings", None), "agent_id", "agent")
        if world_model is not None:
            try:
                world_model.add_visual_observation(
                    agent_id,
                    image=image_array,
                    features=image_features,
                    text=text_payload,
                )
                world_model_updated = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to update world model: %s", exc)

    return {
        "image_path": image_path.as_posix(),
        "description": description,
        "description_style": style,
        "image_features": list(image_features),
        "text_features": list(text_features) if text_features is not None else None,
        "workspace_file": summary_path.relative_to(agent.workspace.root).as_posix(),
        "world_model_updated": world_model_updated,
        "clip_model": clip_handles.used_model,
    }
