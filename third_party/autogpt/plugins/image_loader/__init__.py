"""Simple plugin demonstrating image acquisition.

This plugin exposes a ``load_image`` function that reads image bytes from disk
and returns a :class:`~autogpt.core.multimodal.MultimodalInput` instance. It can
be used by abilities or planners that expect multimodal input.
"""

from pathlib import Path
from typing import Optional

from autogpt.core.multimodal import MultimodalInput


def load_image(path: str, text: str = "", metadata: Optional[dict] = None) -> MultimodalInput:
    """Load an image from ``path`` and wrap it in ``MultimodalInput``.

    Args:
        path: Location of the image file to load.
        text: Optional text associated with the image.
        metadata: Optional metadata to attach to the input.
    """

    data = Path(path).read_bytes()
    return MultimodalInput(text=text, image=data, metadata=metadata or {"source": path})
