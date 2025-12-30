"""Model pruning and quantization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Literal


def optimize_model(
    model_path: str | Path,
    backend: Literal["onnxruntime", "tensorrt"] = "onnxruntime",
    quantize: bool = True,
) -> Path:
    """Optimise a model using pruning and/or quantisation.

    Parameters
    ----------
    model_path:
        Path to the source model file.
    backend:
        Backend to use for optimisation: ``"onnxruntime"`` or ``"tensorrt"``.
    quantize:
        Whether to apply dynamic quantisation when supported.

    Returns
    -------
    Path
        Path to the optimised model file.
    """
    model_path = Path(model_path)
    out_path = model_path.with_suffix(".opt.onnx" if backend == "onnxruntime" else ".opt.plan")

    if backend == "onnxruntime":
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:  # pragma: no cover - dependency not always installed
            raise RuntimeError("onnxruntime is required for ONNX optimisations") from exc

        if quantize:
            quantize_dynamic(model_path, out_path, weight_type=QuantType.QInt8)
        else:
            # Simply validate and copy the model
            ort.InferenceSession(model_path)
            out_path.write_bytes(model_path.read_bytes())
    elif backend == "tensorrt":
        try:
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - dependency not always installed
            raise RuntimeError("tensorrt is required for TensorRT optimisations") from exc

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        config = builder.create_builder_config()
        if quantize:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(out_path, "wb") as f:
            f.write(engine.serialize())
    else:
        raise ValueError(f"Unsupported backend {backend}")

    return out_path
