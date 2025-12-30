"""
感知过程模块

实现将外部刺激转换为神经表示的感知功能。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Iterable
import logging
import numpy as np
import random

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess
from BrainSimulationSystem.models.visual_cortex import (
    VisualCortexModel,
    VisionModelUnavailable,
)
from BrainSimulationSystem.models.auditory_cortex import (
    AuditoryCortexModel,
    AuditoryModelUnavailable,
)
from BrainSimulationSystem.models.somatosensory_cortex import SomatosensoryCortex
from BrainSimulationSystem.models.structured_data_parser import StructuredDataParser

try:  # pragma: no cover - optional fusion dependency
    from modules.brain.multimodal import MultimodalFusionEngine
    _HAS_NATIVE_FUSION = True
except Exception:  # pragma: no cover - fallback when fusion module missing
    _HAS_NATIVE_FUSION = False

    class MultimodalFusionEngine:  # type: ignore[override]
        """Lightweight fallback fusion engine averaging aligned modalities."""

        def fuse_sensory_modalities(self, **modalities: Any) -> np.ndarray:
            if not modalities:
                raise ValueError("at least one modality must be provided")
            vectors = []
            for vector in modalities.values():
                arr = np.asarray(vector, dtype=float).reshape(-1)
                if arr.size == 0:
                    continue
                vectors.append(arr)
            if not vectors:
                raise ValueError("no valid modalities provided")
            min_len = min(vec.size for vec in vectors)
            trimmed = [vec[:min_len] for vec in vectors]
            return np.mean(trimmed, axis=0)


class PerceptionProcess(CognitiveProcess):
    """
    感知过程
    
    将外部刺激转换为神经表示
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化感知过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - input_mapping: 输入到神经元的映射方式
                - normalization: 输入归一化方式
                - noise_level: 噪声水平
        """
        super().__init__(network, params or {})
        self._logger = logging.getLogger(self.__class__.__name__)

        vision_cfg = self.params.get("vision", {})
        self._vision_enabled = bool(vision_cfg.get("enabled", vision_cfg))
        self._vision_cortex: Optional[VisualCortexModel] = None
        self._vision_status: Dict[str, Any] = {}

        if self._vision_enabled:
            try:
                model_config = vision_cfg.get("model", vision_cfg)
                self._vision_cortex = VisualCortexModel(model_config)
                self._vision_status["backend"] = self._vision_cortex.backend
                self._vision_status.update(getattr(self._vision_cortex, "status", {}))
            except VisionModelUnavailable as exc:
                self._logger.warning("Vision cortex backend unavailable: %s", exc)
                self._vision_status["error"] = str(exc)
                self._vision_enabled = False
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to initialize visual cortex: %s", exc)
                self._vision_status["error"] = str(exc)
                self._vision_enabled = False

        auditory_cfg = self.params.get("auditory", {})
        self._auditory_enabled = bool(auditory_cfg.get("enabled", auditory_cfg))
        self._auditory_cortex: Optional[AuditoryCortexModel] = None
        self._auditory_status: Dict[str, Any] = {}

        if self._auditory_enabled:
            try:
                model_config = auditory_cfg.get("model")
                if model_config is None:
                    model_config = {k: v for k, v in auditory_cfg.items() if k != "enabled"}
                self._auditory_cortex = AuditoryCortexModel(model_config)
                self._auditory_status["backend"] = self._auditory_cortex.backend
                self._auditory_status.update(getattr(self._auditory_cortex, "status", {}))
            except AuditoryModelUnavailable as exc:
                self._logger.warning("Auditory cortex backend unavailable: %s", exc)
                self._auditory_status["error"] = str(exc)
                self._auditory_enabled = False
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to initialize auditory cortex: %s", exc)
                self._auditory_status["error"] = str(exc)
                self._auditory_enabled = False

        somato_cfg = self.params.get("somatosensory", {})
        self._somatosensory_enabled = bool(somato_cfg.get("enabled", somato_cfg))
        self._somatosensory_cortex: Optional[SomatosensoryCortex] = None
        self._somatosensory_status: Dict[str, Any] = {}

        if self._somatosensory_enabled:
            try:
                model_config = somato_cfg.get("model")
                if model_config is None:
                    model_config = {k: v for k, v in somato_cfg.items() if k != "enabled"}
                self._somatosensory_cortex = SomatosensoryCortex(model_config)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to initialize somatosensory cortex: %s", exc)
                self._somatosensory_status["error"] = str(exc)
                self._somatosensory_enabled = False

        structured_cfg = self.params.get("structured", {})
        self._structured_enabled = bool(structured_cfg.get("enabled", structured_cfg))
        self._structured_parser: Optional[StructuredDataParser] = None
        self._structured_status: Dict[str, Any] = {}
        if self._structured_enabled:
            try:
                parser_cfg = structured_cfg.get("parser")
                if parser_cfg is None:
                    parser_cfg = {k: v for k, v in structured_cfg.items() if k != "enabled"}
                self._structured_parser = StructuredDataParser(parser_cfg)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to initialize structured data parser: %s", exc)
                self._structured_status["error"] = str(exc)
                self._structured_enabled = False

        fusion_cfg = self.params.get("multimodal_fusion", {})
        requested_fusion = bool(fusion_cfg.get("enabled", fusion_cfg))
        self._fusion_enabled = bool(requested_fusion)
        self._fusion_engine: Optional[MultimodalFusionEngine] = None
        self._fusion_status: Dict[str, Any] = {}
        if self._fusion_enabled:
            try:
                engine = fusion_cfg.get("engine")
                if engine is not None and isinstance(engine, MultimodalFusionEngine):
                    self._fusion_engine = engine
                else:
                    self._fusion_engine = MultimodalFusionEngine()
                if not _HAS_NATIVE_FUSION:
                    self._fusion_status.setdefault(
                        "warning",
                        "Using lightweight fallback fusion engine",
                    )
            except Exception as exc:  # pragma: no cover - optional dependency errors
                self._logger.exception("Failed to initialize fusion engine: %s", exc)
                self._fusion_status["error"] = str(exc)
                self._fusion_engine = None
                self._fusion_enabled = False
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理感知输入
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - sensory_data: 感官数据（如视觉、听觉等）
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        input_mapping = self.params.get("input_mapping", "direct")
        normalization = self.params.get("normalization", "minmax")
        noise_level = self.params.get("noise_level", 0.0)
        
        directives = inputs.get("attention_directives") or {}
        modality_weights: Dict[str, float] = directives.get("modality_weights", {}) if isinstance(directives, dict) else {}

        sensory_data = inputs.get("sensory_data")
        prepared_input: List[float] = []
        neural_activity: Dict[Any, Any] = {}

        if sensory_data:
            normalized_data = self._normalize_input(list(sensory_data), normalization)

            if noise_level > 0:
                normalized_data = self._add_noise(normalized_data, noise_level)

            neural_input = self._map_to_neurons(normalized_data, input_mapping)
            prepared_input = list(neural_input)

            if (
                self.network.input_layer_name
                and self.network.input_layer_name in getattr(self.network, "layers", {})
            ):
                input_layer = self.network.layers[self.network.input_layer_name]
                input_size = min(len(prepared_input), input_layer.size)

                prepared_input = prepared_input[:input_size]
                if len(prepared_input) < input_layer.size:
                    prepared_input.extend([0.0] * (input_layer.size - len(prepared_input)))
            else:
                buffer_size = len(getattr(self.network, "_input_buffer", []) or [])
                if buffer_size:
                    if len(prepared_input) > buffer_size:
                        prepared_input = prepared_input[:buffer_size]
                    elif len(prepared_input) < buffer_size:
                        prepared_input.extend([0.0] * (buffer_size - len(prepared_input)))

            set_input = getattr(self.network, "set_input", None)
            if callable(set_input):
                set_input(prepared_input)

            if self.network.input_layer_name and self.network.input_layer_name in getattr(
                self.network, "layers", {}
            ):
                layer = self.network.layers[self.network.input_layer_name]
                neural_activity = {
                    neuron_id: self.network.neurons[neuron_id].voltage
                    for neuron_id in layer.neuron_ids
                }

        vision_result = self._process_vision(inputs)
        if vision_result is not None:
            self._apply_modality_directive("vision", vision_result, modality_weights)
        auditory_result = self._process_auditory(inputs)
        if auditory_result is not None:
            self._apply_modality_directive("auditory", auditory_result, modality_weights)
        somato_result = self._process_somatosensory(inputs)
        structured_result = self._process_structured(inputs)
        if structured_result is not None:
            self._apply_modality_directive("structured", structured_result, modality_weights)
        fusion_result = self._compute_multimodal_fusion(
            vision_result,
            auditory_result,
            structured_result,
            inputs,
        )

        result: Dict[str, Any] = {
            "perception_output": prepared_input,
            "neural_activity": neural_activity,
        }
        if vision_result is not None:
            result["vision"] = vision_result
        elif self._vision_status.get("error"):
            result["vision"] = {"error": self._vision_status["error"]}

        if auditory_result is not None:
            result["auditory"] = auditory_result
        elif self._auditory_status.get("error"):
            result.setdefault("auditory", {"error": self._auditory_status["error"]})

        if somato_result is not None:
            result["somatosensory"] = somato_result
        elif self._somatosensory_status.get("error"):
            result.setdefault("somatosensory", {"error": self._somatosensory_status["error"]})

        if structured_result is not None:
            result["structured"] = structured_result
        elif self._structured_status.get("error"):
            result.setdefault("structured", {"error": self._structured_status["error"]})

        if fusion_result is not None:
            result["multimodal_fusion"] = fusion_result
        elif self._fusion_status.get("error"):
            result.setdefault("multimodal_fusion", {"error": self._fusion_status["error"]})

        return result

    def _process_auditory(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._auditory_enabled or self._auditory_cortex is None:
            return None

        audio_input: Optional[Any] = None
        sample_rate = inputs.get("audio_sample_rate")

        for key in ("auditory", "audio", "audio_waveform", "audio_signal"):
            if key in inputs and inputs[key] is not None:
                value = inputs[key]
                if isinstance(value, dict):
                    audio_input = value.get("waveform", value.get("data"))
                    sample_rate = value.get("sample_rate", sample_rate)
                else:
                    audio_input = value
                break

        if audio_input is None:
            return None

        try:
            output = self._auditory_cortex.process(audio_input, sample_rate=sample_rate)
            output["backend"] = self._auditory_cortex.backend
            status = getattr(self._auditory_cortex, "status", {})
            if status:
                output.setdefault("status", status)
                self._auditory_status.update(status)
            if getattr(self._auditory_cortex, "advanced_backend_failed", False):
                warning = status.get("warning") if isinstance(status, dict) else None
                if warning:
                    output.setdefault("warning", warning)
                output["confidence"] = min(output.get("confidence", 0.5), 0.45)
            return output
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception("Auditory processing failed: %s", exc)
            self._auditory_status["error"] = str(exc)
            backend = getattr(self._auditory_cortex, "backend", "unknown")
            return {"error": str(exc), "backend": backend}

    def _process_somatosensory(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._somatosensory_enabled or self._somatosensory_cortex is None:
            return None

        data_candidates = (
            inputs.get("somatosensory"),
            inputs.get("tactile"),
            inputs.get("pressure"),
            inputs.get("sensor_data"),
        )
        sensor_data = None
        for candidate in data_candidates:
            if candidate is not None:
                sensor_data = candidate
                break
        if sensor_data is None:
            return None

        try:
            return self._somatosensory_cortex.process(sensor_data)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception("Somatosensory processing failed: %s", exc)
            self._somatosensory_status["error"] = str(exc)
            return {"error": str(exc)}

    def _process_structured(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._structured_enabled or self._structured_parser is None:
            return None

        payload, source = self._extract_structured_payload(inputs)
        if payload is None:
            return None

        try:
            batch = self._structured_parser.parse(payload, source=source)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception("Structured data parsing failed: %s", exc)
            self._structured_status["error"] = str(exc)
            return {"error": str(exc)}

        data = batch.as_dict()
        provenance = source.get("name") or source.get("path")
        if provenance:
            data["provenance"] = provenance
        summary = data.get("embeddings", {}).get("summary")
        if summary is not None:
            data["summary_embedding"] = summary
        return data

    def _apply_modality_directive(
        self,
        modality: str,
        result: Dict[str, Any],
        modality_weights: Dict[str, float],
    ) -> None:
        if not result or not modality_weights:
            return
        weight = modality_weights.get(modality)
        if weight is None:
            return
        if "confidence" in result:
            result["confidence"] = float(
                np.clip(result.get("confidence", 0.0) * (0.5 + weight), 0.0, 1.0)
            )
        result["attention_weight"] = float(weight)

    def _process_vision(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._vision_enabled or self._vision_cortex is None:
            return None

        image_input: Optional[Any] = None
        for key in ("vision", "image", "vision_image"):
            if key in inputs and inputs[key] is not None:
                image_input = inputs[key]
                break
        if image_input is None:
            return None

        vision_cfg = self.params.get("vision", {})
        return_feature_maps = vision_cfg.get("return_feature_maps", True)

        try:
            vision_output = self._vision_cortex.process(
                image_input,
                return_feature_maps=return_feature_maps,
            )
            vision_output["backend"] = self._vision_cortex.backend
            status = getattr(self._vision_cortex, "status", {})
            if status:
                vision_output.setdefault("status", status)
                self._vision_status.update(status)
            if getattr(self._vision_cortex, "advanced_backend_failed", False):
                warning = status.get("warning") if isinstance(status, dict) else None
                if warning:
                    vision_output.setdefault("warning", warning)
                vision_output["confidence"] = min(vision_output.get("confidence", 0.6), 0.4)
            return vision_output
        except Exception as exc:  # pragma: no cover - vision失败不应终止主流程
            self._logger.exception("Vision processing failed: %s", exc)
            self._vision_status["error"] = str(exc)
            return {"error": str(exc), "backend": self._vision_cortex.backend}
    
    def _compute_multimodal_fusion(
        self,
        vision_result: Optional[Dict[str, Any]],
        auditory_result: Optional[Dict[str, Any]],
        structured_result: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._fusion_enabled or self._fusion_engine is None:
            return None

        directives = inputs.get("attention_directives") or {}
        modality_weights: Dict[str, float] = (
            directives.get("modality_weights", {}) if isinstance(directives, dict) else {}
        )

        modalities: Dict[str, Any] = {}
        if vision_result and "embedding" in vision_result:
            vector = self._to_numpy_vector(vision_result["embedding"])
            if vector is not None:
                modalities["vision"] = {
                    "embedding": vector,
                    "metadata": {
                        "confidence": vision_result.get("confidence"),
                        "weight": vision_result.get("attention_weight", modality_weights.get("vision", 1.0)),
                    },
                }
        if auditory_result and "embedding" in auditory_result:
            vector = self._to_numpy_vector(auditory_result["embedding"])
            if vector is not None:
                modalities["auditory"] = {
                    "embedding": vector,
                    "metadata": {
                        "confidence": auditory_result.get("confidence"),
                        "weight": auditory_result.get("attention_weight", modality_weights.get("auditory", 1.0)),
                    },
                }
        if structured_result:
            embeddings = structured_result.get("embeddings") or {}
            summary = structured_result.get("summary_embedding") or embeddings.get("summary")
            vector = self._to_numpy_vector(summary)
            if vector is not None:
                modalities["structured"] = {
                    "embedding": vector,
                    "metadata": {
                        "confidence": structured_result.get("confidence"),
                        "weight": structured_result.get("attention_weight", modality_weights.get("structured", 1.0)),
                    },
                }

        language_vector = self._language_embedding_from_inputs(inputs)
        if language_vector is not None:
            modalities["language"] = {
                "embedding": language_vector,
                "metadata": {
                    "confidence": inputs.get("language_confidence"),
                    "weight": modality_weights.get("language", 1.0),
                },
            }

        if len(modalities) < 2:
            return None

        try:
            fused = self._fusion_engine.fuse_sensory_modalities(**modalities)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception("Multimodal fusion failed: %s", exc)
            self._fusion_status["error"] = str(exc)
            return {"error": str(exc)}

        attention_weights: Dict[str, float] = {}
        try:
            scores = []
            names = []
            for name, payload in modalities.items():
                if not isinstance(payload, dict):
                    continue
                vector = payload.get("embedding")
                meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                arr = self._to_numpy_vector(vector)
                if arr is None:
                    continue
                base = float(np.linalg.norm(arr))
                try:
                    confidence = float(meta.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                try:
                    weight_hint = float(meta.get("weight", 1.0))
                except (TypeError, ValueError):
                    weight_hint = 1.0
                score = base * max(0.0, confidence) * max(0.0, weight_hint)
                scores.append(score)
                names.append(str(name))

            if scores:
                weights = np.asarray(scores, dtype=float)
                if np.allclose(weights, 0.0):
                    weights = np.ones_like(weights)
                weights = weights - float(np.max(weights))
                weights = np.exp(weights)
                weights = weights / float(np.sum(weights))
                attention_weights = {names[i]: float(weights[i]) for i in range(len(names))}
        except Exception:  # pragma: no cover - best-effort diagnostics only
            attention_weights = {}

        return {
            "embedding": fused.tolist(),
            "modalities": list(modalities.keys()),
            "attention_weights": attention_weights,
        }

    def _language_embedding_from_inputs(self, inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("language_embedding", "language_vector", "text_embedding"):
            if key in inputs:
                vector = self._to_numpy_vector(inputs.get(key))
                if vector is not None:
                    return vector

        language_payload = inputs.get("language_input")
        if isinstance(language_payload, dict):
            for key in ("embedding", "vector", "features"):
                if key in language_payload:
                    vector = self._to_numpy_vector(language_payload.get(key))
                    if vector is not None:
                        return vector
        return None

    def _extract_structured_payload(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        keys = (
            "structured_data",
            "structured",
            "table",
            "knowledge",
            "structured_payload",
        )
        candidate = None
        for key in keys:
            if key in inputs and inputs[key] is not None:
                candidate = inputs[key]
                break
        if candidate is None:
            return None, {}

        payload = candidate
        source: Dict[str, Any] = {}
        if isinstance(candidate, dict):
            for key in ("payload", "data", "records", "rows", "table"):
                if key in candidate and candidate[key] is not None:
                    payload = candidate[key]
                    break
            if payload is None and "triples" in candidate:
                payload = candidate["triples"]
            if isinstance(candidate.get("source"), dict):
                source.update(candidate["source"])
            for meta_key in (
                "format",
                "path",
                "name",
                "header",
                "relations",
                "subject_field",
                "object_field",
            ):
                if meta_key in candidate and meta_key not in source:
                    source[meta_key] = candidate[meta_key]

        extra_source = inputs.get("structured_source")
        if isinstance(extra_source, dict):
            source.update({k: v for k, v in extra_source.items() if v is not None})
        if payload is None:
            payload = candidate
        return payload, source

    @staticmethod
    def _to_numpy_vector(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.size == 0:
            return None
        if array.ndim == 0:
            array = array.reshape(1)
        return array.reshape(-1)

    def _normalize_input(self, data: List[float], method: str) -> List[float]:
        """
        归一化输入数据
        
        Args:
            data: 输入数据
            method: 归一化方法
            
        Returns:
            归一化后的数据
        """
        if not data:
            return []
        
        if method == "minmax":
            # Min-Max归一化
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                return [0.5] * len(data)
            return [(x - min_val) / (max_val - min_val) for x in data]
        
        elif method == "zscore":
            # Z-score归一化
            mean = sum(data) / len(data)
            std = np.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
            if std == 0:
                return [0.0] * len(data)
            return [(x - mean) / std for x in data]
        
        elif method == "sigmoid":
            # Sigmoid归一化
            return [1.0 / (1.0 + np.exp(-x)) for x in data]
        
        else:
            # 默认不做归一化
            return data
    
    def _add_noise(self, data: List[float], noise_level: float) -> List[float]:
        """
        添加噪声
        
        Args:
            data: 输入数据
            noise_level: 噪声水平
            
        Returns:
            添加噪声后的数据
        """
        return [x + random.uniform(-noise_level, noise_level) for x in data]
    
    def _map_to_neurons(self, data: List[float], mapping: str) -> List[float]:
        """
        将数据映射到神经元输入
        
        Args:
            data: 输入数据
            mapping: 映射方法
            
        Returns:
            神经元输入
        """
        if mapping == "direct":
            # 直接映射
            return data
        
        elif mapping == "population":
            # 群体编码
            result = []
            for x in data:
                # 为每个值创建一个小型群体编码
                population_size = self.params.get("population_size", 5)
                mean = x
                std = self.params.get("population_std", 0.1)
                population = [random.normalvariate(mean, std) for _ in range(population_size)]
                result.extend(population)
            return result
        
        elif mapping == "sparse":
            # 稀疏编码
            sparsity = self.params.get("sparsity", 0.1)
            size = len(data) * 10  # 扩大输出大小
            result = [0.0] * size
            
            for i, x in enumerate(data):
                # 确定激活的神经元数量
                active_count = max(1, int(size * sparsity / len(data)))
                
                # 随机选择神经元激活
                indices = random.sample(range(i * 10, (i + 1) * 10), active_count)
                for idx in indices:
                    if idx < size:
                        result[idx] = x
            
            return result
        
        else:
            # 默认直接映射
            return data

