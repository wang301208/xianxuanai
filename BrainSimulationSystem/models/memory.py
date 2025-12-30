"""
完整记忆系统实现 - 基础模块
Complete Memory System Implementation - Base Module

实现生物学真实的记忆机制基础组件
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import time
import json
from collections import defaultdict, deque
import hashlib

from BrainSimulationSystem.models.cognitive_base import CognitiveProcess
from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.memory.vector_store import VectorMemoryStore

class MemoryType(Enum):
    """记忆类型"""
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    IMPLICIT = "implicit"

class MemoryPhase(Enum):
    """记忆阶段"""
    ENCODING = "encoding"
    CONSOLIDATION = "consolidation"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    RECONSOLIDATION = "reconsolidation"
    FORGETTING = "forgetting"

class ConsolidationState(Enum):
    """巩固状态"""
    LABILE = "labile"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    RECONSOLIDATING = "reconsolidating"

@dataclass
class MemoryTrace:
    """记忆痕迹"""
    trace_id: str
    content: Any
    memory_type: MemoryType
    encoding_time: float
    last_access_time: float
    access_count: int = 0
    
    # 记忆强度和稳定性
    strength: float = 1.0
    stability: float = 0.5
    retrievability: float = 1.0
    
    # 巩固状态
    consolidation_state: ConsolidationState = ConsolidationState.LABILE
    consolidation_progress: float = 0.0
    
    # 关联信息
    associations: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # 神经基础
    neural_pattern: Optional[np.ndarray] = None
    brain_regions: List[str] = field(default_factory=list)
    
    # 情感标记
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    
    def update_strength(self, dt: float, rehearsal: bool = False, 
                       interference: float = 0.0) -> float:
        """更新记忆强度"""
        
        # 遗忘曲线（Ebbinghaus）
        forgetting_rate = 0.001 * (1.0 - self.stability)
        decay = self.strength * forgetting_rate * dt
        
        # 干扰效应
        interference_decay = interference * 0.01 * dt
        
        # 复述增强
        rehearsal_boost = 0.1 if rehearsal else 0.0
        
        # 更新强度
        self.strength += rehearsal_boost - decay - interference_decay
        self.strength = np.clip(self.strength, 0.0, 2.0)
        
        # 更新可提取性
        self.retrievability = self.strength * (1.0 - interference * 0.5)
        self.retrievability = np.clip(self.retrievability, 0.0, 1.0)
        
        return self.strength
    
    def update_consolidation(self, dt: float, consolidation_signal: float = 0.0):
        """更新巩固状态"""
        
        if self.consolidation_state == ConsolidationState.LABILE:
            if consolidation_signal > 0.5:
                self.consolidation_state = ConsolidationState.CONSOLIDATING
                self.consolidation_progress = 0.1
        
        elif self.consolidation_state == ConsolidationState.CONSOLIDATING:
            consolidation_rate = 0.001 * consolidation_signal
            self.consolidation_progress += consolidation_rate * dt
            
            if self.consolidation_progress >= 1.0:
                self.consolidation_state = ConsolidationState.CONSOLIDATED
                self.stability += 0.3
                self.stability = min(self.stability, 1.0)
        
        elif self.consolidation_state == ConsolidationState.CONSOLIDATED:
            if self.access_count > 0 and consolidation_signal > 0.7:
                self.consolidation_state = ConsolidationState.RECONSOLIDATING
                self.consolidation_progress = 0.5
        
        elif self.consolidation_state == ConsolidationState.RECONSOLIDATING:
            reconsolidation_rate = 0.002 * consolidation_signal
            self.consolidation_progress += reconsolidation_rate * dt
            
            if self.consolidation_progress >= 1.0:
                self.consolidation_state = ConsolidationState.CONSOLIDATED
                self.stability += 0.1
                self.stability = min(self.stability, 1.0)
    
    def access(self, current_time: float) -> float:
        """访问记忆"""
        
        self.last_access_time = current_time
        self.access_count += 1
        
        # 测试效应：提取增强记忆
        retrieval_boost = 0.05 * min(self.access_count, 10)
        self.strength += retrieval_boost
        self.strength = min(self.strength, 2.0)
        
        return self.retrievability
    
    def add_association(self, other_trace_id: str, association_strength: float):
        """添加关联"""
        self.associations[other_trace_id] = association_strength
    
    def get_memory_age(self, current_time: float) -> float:
        """获取记忆年龄"""
        return current_time - self.encoding_time

@dataclass
class WorkingMemoryBuffer:
    """工作记忆缓冲区"""
    capacity: int = 7  # Miller's magic number ± 2
    decay_rate: float = 0.1  # per second
    
    # 缓冲区内容
    items: List[Dict[str, Any]] = field(default_factory=list)
    attention_weights: List[float] = field(default_factory=list)
    
    def add_item(self, item: Dict[str, Any], attention_weight: float = 1.0) -> bool:
        """添加项目到工作记忆"""
        
        if len(self.items) >= self.capacity:
            # 移除最弱的项目
            min_idx = np.argmin(self.attention_weights)
            self.items.pop(min_idx)
            self.attention_weights.pop(min_idx)
        
        self.items.append(item)
        self.attention_weights.append(attention_weight)
        
        return True
    
    def update(self, dt: float, rehearsal_indices: List[int] = None) -> List[Dict[str, Any]]:
        """更新工作记忆"""
        
        if rehearsal_indices is None:
            rehearsal_indices = []
        
        # 衰减所有项目
        for i in range(len(self.attention_weights)):
            if i not in rehearsal_indices:
                self.attention_weights[i] *= np.exp(-self.decay_rate * dt)
        
        # 移除衰减过度的项目
        items_to_remove = []
        for i, weight in enumerate(self.attention_weights):
            if weight < 0.1:
                items_to_remove.append(i)
        
        # 从后往前删除，避免索引问题
        for i in reversed(items_to_remove):
            self.items.pop(i)
            self.attention_weights.pop(i)
        
        return self.items.copy()
    
    def retrieve_item(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检索项目"""
        
        best_match = None
        best_score = 0.0
        
        for i, item in enumerate(self.items):
            score = self._calculate_match_score(item, query)
            score *= self.attention_weights[i]
            
            if score > best_score:
                best_score = score
                best_match = item
        
        return best_match if best_score > 0.5 else None
    
    def _calculate_match_score(self, item: Dict[str, Any], query: Dict[str, Any]) -> float:
        """计算匹配分数"""
        
        if not query:
            return 0.0
        
        matches = 0
        total_keys = len(query)
        
        for key, value in query.items():
            if key in item and item[key] == value:
                matches += 1
        
        return matches / total_keys if total_keys > 0 else 0.0


@dataclass
class IndexedWorkingMemoryBuffer:
    """Ring/index working memory buffer with lightweight gating/eviction.

    Keeps the same public API as ``WorkingMemoryBuffer`` (`add_item`, `update`,
    `retrieve_item`, plus `items` and `attention_weights`) so it can be swapped
    in via config without touching downstream modules.
    """

    capacity: int = 32
    decay_rate: float = 0.1
    min_weight: float = 0.1
    eviction_policy: str = "lowest_weight"  # "lowest_weight" | "oldest"
    key_fields: Tuple[str, ...] = ("id", "concept", "event", "action", "text", "query", "name")

    _entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _order: deque = field(default_factory=deque)

    @property
    def items(self) -> List[Any]:
        return [self._entries[key]["item"] for key in self._order if key in self._entries]

    @property
    def attention_weights(self) -> List[float]:
        weights: List[float] = []
        for key in self._order:
            entry = self._entries.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                weights.append(float(entry.get("weight", 0.0)))
            except (TypeError, ValueError):
                weights.append(0.0)
        return weights

    def _make_key(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for field_name in self.key_fields:
                if field_name in payload and payload[field_name] not in (None, "", []):
                    return f"{field_name}:{payload[field_name]}"
            try:
                blob = json.dumps(payload, sort_keys=True, default=str)
            except Exception:
                blob = str(payload)
            return hashlib.md5(blob.encode("utf-8")).hexdigest()
        return hashlib.md5(str(payload).encode("utf-8")).hexdigest()

    def _evict_one(self) -> None:
        if not self._order:
            return
        policy = str(self.eviction_policy or "lowest_weight").lower()
        if policy == "oldest":
            victim = self._order.popleft()
            self._entries.pop(victim, None)
            return

        # Default: evict the weakest item.
        victim = None
        victim_weight = float("inf")
        for key in list(self._order):
            entry = self._entries.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                weight = float(entry.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight < victim_weight:
                victim_weight = weight
                victim = key
        if victim is None:
            victim = self._order.popleft()
        else:
            try:
                self._order.remove(victim)
            except ValueError:
                pass
        self._entries.pop(victim, None)

    def add_item(self, item: Any, attention_weight: float = 1.0) -> bool:
        if self.capacity <= 0:
            return False

        key = self._make_key(item)
        try:
            weight_value = float(attention_weight)
        except (TypeError, ValueError):
            weight_value = 1.0

        if key in self._entries:
            entry = self._entries[key]
            entry["item"] = item
            try:
                entry["weight"] = float(max(float(entry.get("weight", 0.0)), weight_value))
            except Exception:
                entry["weight"] = weight_value
            entry["last_access_time"] = time.time()
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return True

        while len(self._order) >= int(self.capacity):
            self._evict_one()

        self._entries[key] = {
            "item": item,
            "weight": weight_value,
            "encoding_time": time.time(),
            "last_access_time": time.time(),
        }
        self._order.append(key)
        return True

    def update(self, dt: float, rehearsal_indices: List[int] = None) -> List[Any]:
        if rehearsal_indices is None:
            rehearsal_indices = []

        dt_value = float(dt) if isinstance(dt, (int, float, np.floating)) else 0.0
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            dt_value = 0.0

        rehearsal_keys = set()
        for idx in rehearsal_indices:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < len(self._order):
                rehearsal_keys.add(list(self._order)[idx_int])

        for key in list(self._order):
            if key in rehearsal_keys:
                continue
            entry = self._entries.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                entry["weight"] = float(entry.get("weight", 0.0)) * float(np.exp(-self.decay_rate * dt_value))
            except Exception:
                entry["weight"] = 0.0

        # Drop items below threshold.
        for key in list(self._order):
            entry = self._entries.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                weight = float(entry.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight < float(self.min_weight):
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                self._entries.pop(key, None)

        return self.items

    @staticmethod
    def _calculate_match_score(item: Any, query: Dict[str, Any]) -> float:
        if not query or not isinstance(item, dict):
            return 0.0
        matches = 0
        total_keys = len(query)
        for key, value in query.items():
            if key in item and item[key] == value:
                matches += 1
        return matches / total_keys if total_keys > 0 else 0.0

    def retrieve_item(self, query: Dict[str, Any]) -> Optional[Any]:
        key = self._make_key(query)
        entry = self._entries.get(key)
        if isinstance(entry, dict):
            entry["last_access_time"] = time.time()
            return entry.get("item")

        best_match = None
        best_score = 0.0
        for idx, candidate in enumerate(self.items):
            score = self._calculate_match_score(candidate, query)
            try:
                score *= float(self.attention_weights[idx])
            except Exception:
                pass
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match if best_score > 0.5 else None


class ComprehensiveMemorySystem:
    """综合记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 导入子系统
        from .hippocampal_memory import HippocampalMemorySystem
        from .neocortical_memory import NeocorticalMemorySystem
        
        # 子系统
        self.hippocampal_system = HippocampalMemorySystem(
            config.get('hippocampal', {})
        )
        self.neocortical_system = NeocorticalMemorySystem(
            config.get('neocortical', {})
        )
        
        # 工作记忆：支持可选的环形缓冲/索引实现以提升容量与检索效率
        wm_cfg = config.get("working_memory", {})
        if not isinstance(wm_cfg, dict):
            wm_cfg = {}
        wm_strategy = str(wm_cfg.get("strategy", config.get("working_memory_strategy", "priority"))).lower()
        try:
            wm_capacity = int(wm_cfg.get("capacity", config.get("working_memory_capacity", 7)))
        except (TypeError, ValueError):
            wm_capacity = 7
        if wm_capacity <= 0:
            wm_capacity = 7
        try:
            wm_decay = float(wm_cfg.get("decay_rate", 0.1))
        except (TypeError, ValueError):
            wm_decay = 0.1

        if wm_strategy in {"indexed", "indexed_ring", "indexed_buffer"}:
            try:
                min_weight = float(wm_cfg.get("min_weight", 0.1))
            except (TypeError, ValueError):
                min_weight = 0.1
            self.working_memory = IndexedWorkingMemoryBuffer(
                capacity=wm_capacity,
                decay_rate=wm_decay,
                min_weight=min_weight,
                eviction_policy=str(wm_cfg.get("eviction_policy", "lowest_weight")),
            )
        else:
            self.working_memory = WorkingMemoryBuffer(
                capacity=wm_capacity,
                decay_rate=wm_decay,
            )
        
        # 感觉记忆
        self.sensory_buffers = {
            'visual': deque(maxlen=100),
            'auditory': deque(maxlen=50),
            'tactile': deque(maxlen=30)
        }
        
        # 记忆管理
        self.memory_registry = {}
        self.forgetting_scheduler = {}
        
        # 系统间交互
        self.hippocampal_neocortical_transfer = {}
        self.consolidation_queue = deque()
        
        # 性能监控
        self.memory_stats = {
            'total_memories': 0,
            'encoding_rate': 0.0,
            'retrieval_success_rate': 0.0,
            'consolidation_rate': 0.0
        }
        
        self.logger = logging.getLogger("ComprehensiveMemorySystem")
        semantic_cfg = config.get("semantic_bridge", {})
        if not isinstance(semantic_cfg, dict):
            semantic_cfg = {}
        self.semantic_bridge_enabled = bool(semantic_cfg.get("enabled", False))
        try:
            self.semantic_bridge_max_associations = int(semantic_cfg.get("max_associations", 16))
        except (TypeError, ValueError):
            self.semantic_bridge_max_associations = 16
        if self.semantic_bridge_max_associations <= 0:
            self.semantic_bridge_max_associations = 16
        try:
            self.semantic_bridge_relation_strength = float(semantic_cfg.get("relation_strength", 0.4))
        except (TypeError, ValueError):
            self.semantic_bridge_relation_strength = 0.4
        self.semantic_network = None

    def attach_semantic_network(self, semantic_network: Any) -> None:
        """Attach the language ``SemanticNetwork`` (or compatible) for semantic memory updates."""

        self.semantic_network = semantic_network
        attach = getattr(self.neocortical_system, "attach_symbolic_network", None)
        if callable(attach):
            try:
                attach(semantic_network)
            except Exception:
                pass
        else:
            try:
                setattr(self.neocortical_system, "symbolic_network", semantic_network)
            except Exception:
                pass

        # Attaching a semantic network implies we want concept bridging.
        self.semantic_bridge_enabled = True

    def _extract_semantic_payload(
        self,
        content: Any,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """Extract (concept, properties, associations) from an episodic payload."""

        def _add_assoc(target: List[str], value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _add_assoc(target, item)
                return
            text = str(value).strip()
            if text:
                target.append(text)

        associations: List[str] = []
        properties: Dict[str, Any] = {}

        concept: Any = None
        if isinstance(content, dict):
            concept = (
                content.get("concept")
                or content.get("event")
                or content.get("title")
                or content.get("name")
            )
            if concept is None and isinstance(context, dict):
                directives = context.get("attention_directives")
                if isinstance(directives, dict):
                    concept = directives.get("semantic_focus")
                concept = concept or context.get("semantic_focus")
            concept = concept or "episode"

            reserved = {
                "concept",
                "event",
                "title",
                "name",
                "associations",
                "entities",
                "tags",
                "objects",
                "participants",
                "agents",
            }
            for key, value in content.items():
                if key in reserved:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    properties[str(key)] = value

            for key in ("associations", "entities", "tags", "objects", "participants", "agents"):
                if key in content:
                    _add_assoc(associations, content.get(key))
            for key in ("agent", "subject", "object", "action", "location"):
                if key in content:
                    _add_assoc(associations, content.get(key))
        else:
            concept = str(content or "episode").strip()[:64] or "episode"

        if isinstance(context, dict):
            directives = context.get("attention_directives")
            if isinstance(directives, dict):
                _add_assoc(associations, directives.get("semantic_focus"))
            _add_assoc(associations, context.get("tags"))

        # Dedupe + cap list size.
        deduped: List[str] = []
        seen = set()
        for item in associations:
            key = str(item).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(item).strip())
            if len(deduped) >= int(self.semantic_bridge_max_associations):
                break

        return str(concept), properties, deduped

    def _bridge_episode_to_semantic(self, content: Any, context: Optional[Dict[str, Any]]) -> None:
        concept, properties, associations = self._extract_semantic_payload(content, context)
        if not concept:
            return
        self.neocortical_system.encode_semantic_memory(concept, properties, associations)

        # Mirror explicit episodic co-occurrence into the symbolic semantic network
        # (when attached) using a configurable baseline strength. This avoids
        # relying on embedding similarity thresholds for the bridge.
        network = self.semantic_network
        add_node = getattr(network, "add_node", None) if network is not None else None
        add_relation = getattr(network, "add_relation", None) if network is not None else None
        if not callable(add_node) or not callable(add_relation):
            return

        concept_key = str(concept).strip().lower()
        if not concept_key:
            return

        try:
            add_node(concept_key, {"source": "episodic_bridge"})
        except Exception:
            return

        try:
            strength = float(self.semantic_bridge_relation_strength)
        except (TypeError, ValueError):
            strength = 0.4

        for assoc in associations:
            assoc_key = str(assoc).strip().lower()
            if not assoc_key or assoc_key == concept_key:
                continue
            try:
                add_node(assoc_key, {"source": "episodic_bridge"})
                add_relation(concept_key, assoc_key, "episodic_association", strength=strength)
                add_relation(assoc_key, concept_key, "episodic_association", strength=strength)
            except Exception:
                continue
    
    async def encode_memory(self, content: Any, memory_type: MemoryType,
                          context: Dict[str, Any] = None,
                          emotional_tags: Dict[str, float] = None) -> str:
        """编码记忆"""
        
        if context is None:
            context = {}
        if emotional_tags is None:
            emotional_tags = {'valence': 0.0, 'arousal': 0.0}
        
        memory_id = None
        
        try:
            if memory_type == MemoryType.EPISODIC:
                memory_id = self.hippocampal_system.encode_episodic_memory(
                    content, context
                )
                if self.semantic_bridge_enabled or self.semantic_network is not None:
                    try:
                        self._bridge_episode_to_semantic(content, context)
                    except Exception:
                        pass
                
            elif memory_type == MemoryType.SEMANTIC:
                if isinstance(content, dict):
                    concept = content.get('concept', 'unknown')
                    properties = content.get('properties', {})
                    associations = content.get('associations', [])
                else:
                    concept = str(content or 'unknown')
                    properties = {}
                    associations = []
                
                memory_id = self.neocortical_system.encode_semantic_memory(
                    concept, properties, associations
                )
                
            elif memory_type == MemoryType.PROCEDURAL:
                if isinstance(content, dict):
                    skill_name = content.get('skill_name', 'unknown_skill')
                    procedure = content.get('procedure', [])
                else:
                    skill_name = str(content or 'unknown_skill')
                    procedure = []
                
                memory_id = self.neocortical_system.encode_procedural_memory(
                    skill_name, procedure
                )
                
            elif memory_type == MemoryType.WORKING:
                attention_weight = context.get('attention_weight', 1.0)
                success = self.working_memory.add_item(content, attention_weight)
                memory_id = f"wm_{int(time.time())}" if success else None
                
            elif memory_type in [MemoryType.SENSORY]:
                modality = context.get('modality', 'visual')
                if modality in self.sensory_buffers:
                    self.sensory_buffers[modality].append({
                        'content': content,
                        'timestamp': time.time(),
                        'context': context
                    })
                    memory_id = f"sensory_{modality}_{int(time.time())}"
            
            # 注册记忆
            if memory_id:
                self.memory_registry[memory_id] = {
                    'memory_type': memory_type,
                    'encoding_time': time.time(),
                    'emotional_tags': emotional_tags,
                    'context': context,
                    'system': self._get_memory_system(memory_type)
                }
                
                self.memory_stats['total_memories'] += 1
                self.logger.debug(f"Encoded {memory_type.value} memory: {memory_id}")
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Memory encoding failed: {e}")
            return None
    
    def _get_memory_system(self, memory_type: MemoryType) -> str:
        """获取记忆系统名称"""
        
        if memory_type in [MemoryType.EPISODIC]:
            return 'hippocampal'
        elif memory_type in [MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            return 'neocortical'
        elif memory_type == MemoryType.WORKING:
            return 'working'
        else:
            return 'sensory'
    
    async def retrieve_memory(self, query: Dict[str, Any], 
                            memory_type: MemoryType = None,
                            context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """检索记忆"""
        
        results = []
        
        try:
            if memory_type is None or memory_type == MemoryType.EPISODIC:
                episodic_result = self.hippocampal_system.retrieve_episodic_memory(
                    query, context
                )
                if episodic_result:
                    results.append({
                        'memory_type': MemoryType.EPISODIC,
                        'content': episodic_result.content,
                        'memory_id': episodic_result.trace_id,
                        'strength': episodic_result.strength,
                        'context': episodic_result.context
                    })
            
            if memory_type is None or memory_type == MemoryType.SEMANTIC:
                query_text = query.get('concept', query.get('query', ''))
                semantic_results = self.neocortical_system.retrieve_semantic_memory(
                    query_text, context
                )
                
                for result in semantic_results:
                    results.append({
                        'memory_type': MemoryType.SEMANTIC,
                        'content': {
                            'concept': result['concept'],
                            'properties': result['properties']
                        },
                        'memory_id': result['concept_id'],
                        'similarity': result['similarity']
                    })
            
            if memory_type is None or memory_type == MemoryType.WORKING:
                wm_result = self.working_memory.retrieve_item(query)
                if wm_result:
                    results.append({
                        'memory_type': MemoryType.WORKING,
                        'content': wm_result,
                        'memory_id': f"wm_retrieved_{int(time.time())}"
                    })
            
            # 更新检索统计
            if results:
                self.memory_stats['retrieval_success_rate'] = (
                    self.memory_stats['retrieval_success_rate'] * 0.9 + 0.1
                )
            else:
                self.memory_stats['retrieval_success_rate'] *= 0.9
            
            self.logger.debug(f"Retrieved {len(results)} memories for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Memory retrieval failed: {e}")
            return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        
        return {
            'total_memories': self.memory_stats['total_memories'],
            'working_memory_items': len(self.working_memory.items),
            'hippocampal_memories': len(self.hippocampal_system.ca1_memories),
            'semantic_concepts': len(self.neocortical_system.concept_embeddings),
            'procedural_skills': len(self.neocortical_system.procedural_memories),
            'retrieval_success_rate': self.memory_stats['retrieval_success_rate'],
            'consolidation_rate': self.memory_stats['consolidation_rate']
        }

    async def update_memory_system(self, dt: float, sleep_mode: bool = False) -> Dict[str, Any]:
        """更新记忆系统（用于完整仿真步进）"""

        # 轻量级实现：维护统计、队列与简单巩固逻辑，保证接口稳定与测试可用。
        try:
            enabled = bool(self.config.get("memory_consolidation", True))
            if sleep_mode or enabled:
                key = "sleep_consolidation_max_items" if sleep_mode else "consolidation_max_items"
                try:
                    max_items = int(self.config.get(key, 64 if sleep_mode else 16))
                except Exception:
                    max_items = 64 if sleep_mode else 16
                await self._run_consolidation_pass(max_items=max(0, min(max_items, 512)))
        except Exception as exc:
            self.logger.warning("Memory consolidation pass skipped: %s", exc)

        try:
            consolidator = getattr(self.hippocampal_system, "consolidate_memories", None)
            if callable(consolidator):
                consolidator(dt, sleep_mode=bool(sleep_mode))
        except Exception as exc:
            self.logger.warning("Hippocampal consolidation skipped: %s", exc)

        try:
            self._run_forgetting_pass()
        except Exception as exc:
            self.logger.warning("Memory forgetting pass skipped: %s", exc)

        # Working memory decay + semantic memory forgetting/capacity maintenance.
        try:
            updater = getattr(self.working_memory, "update", None)
            if callable(updater):
                updater(dt)
        except Exception as exc:
            self.logger.warning("Working memory update skipped: %s", exc)

        try:
            neocortical_update = getattr(self.neocortical_system, "update", None)
            if callable(neocortical_update):
                rehearsal_concepts = None
                getter = getattr(self.semantic_network, "get_most_activated", None) if self.semantic_network is not None else None
                if callable(getter):
                    try:
                        threshold = float(self.config.get("semantic_rehearsal_threshold", 0.3))
                    except (TypeError, ValueError):
                        threshold = 0.3
                    try:
                        rehearsal_concepts = getter(threshold=threshold)
                    except TypeError:
                        rehearsal_concepts = getter(threshold)

                try:
                    if rehearsal_concepts is not None:
                        neocortical_update(dt, rehearsal_concepts=rehearsal_concepts)
                    else:
                        neocortical_update(dt)
                except TypeError:
                    neocortical_update(dt)
        except Exception as exc:
            self.logger.warning("Neocortical memory update skipped: %s", exc)

        # 统计的渐进更新：用 dt 作为平滑权重来源
        alpha = min(0.1, max(0.01, float(dt) / 100.0)) if isinstance(dt, (int, float)) else 0.05
        self.memory_stats["encoding_rate"] = (1.0 - alpha) * float(self.memory_stats.get("encoding_rate", 0.0))
        self.memory_stats["consolidation_rate"] = (1.0 - alpha) * float(self.memory_stats.get("consolidation_rate", 0.0))

        return self.get_memory_statistics()

    async def _run_consolidation_pass(self, max_items: int = 16) -> None:
        """将部分海马记忆转移到新皮层（简化巩固）。"""

        if max_items <= 0:
            return

        transferred = 0

        # 简化策略：从 CA1 中抽样若干 episodic 记忆，提取可序列化字段写入 semantic。
        for trace_id, trace in list(self.hippocampal_system.ca1_memories.items()):
            if transferred >= max_items:
                break
            try:
                content = getattr(trace, "content", None)
                if not isinstance(content, dict):
                    continue
                self._bridge_episode_to_semantic(content, context=None)
                transferred += 1
            except Exception:
                continue

        if transferred:
            self.memory_stats["consolidation_rate"] = float(self.memory_stats.get("consolidation_rate", 0.0)) * 0.9 + 0.1

    def _run_forgetting_pass(self) -> None:
        """轻量遗忘策略：清理 registry 中无效条目。"""

        to_remove = []
        for memory_id, meta in self.memory_registry.items():
            system_name = meta.get("system")
            memory_type = meta.get("memory_type")
            # working memory 随队列自然淘汰；这里仅清理已被覆盖/无效的 registry
            if system_name == "hippocampal":
                if memory_id not in self.hippocampal_system.ca1_memories:
                    to_remove.append(memory_id)
            elif system_name == "neocortical":
                # semantic/procedural 的持久化存储结构不同，这里只做最小清理
                if memory_type == MemoryType.PROCEDURAL and memory_id not in self.neocortical_system.procedural_memories:
                    to_remove.append(memory_id)
            else:
                # sensory/working 不维护强一致 registry
                pass

        for memory_id in to_remove:
            self.memory_registry.pop(memory_id, None)

# 工厂函数


class MemoryProcess(CognitiveProcess):
    """Adapter exposing the comprehensive memory system via a synchronous process API."""

    def __init__(self, network: NeuralNetwork, params: Optional[Dict[str, Any]] = None):
        super().__init__(network, params or {})
        config = params or {}
        system_config = config.get("system", config)
        self.memory_system = ComprehensiveMemorySystem(system_config)
        self.default_memory_type = config.get("default_memory_type", "working")
        self.logger = logging.getLogger(self.__class__.__name__)
        vector_cfg = config.get("vector_store", config.get("vector_memory", {}))
        self.vector_store = VectorMemoryStore(vector_cfg, logger=self.logger)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "stored_id": None,
            "retrieved": [],
            "statistics": {},
        }
        vector_matches: List[Dict[str, Any]] = []
        vector_query_text: Optional[str] = None

        try:
            store_payload = inputs.get("store")
            if store_payload is not None:
                store_items: List[Any]
                if isinstance(store_payload, (list, tuple)):
                    store_items = list(store_payload)
                else:
                    store_items = [store_payload]

                stored_ids: List[str] = []
                for item in store_items:
                    memory_type, content, context, emotion_tags = self._parse_store_payload(item, inputs)
                    if content is None:
                        continue
                    memory_id = self._run_async(
                        self.memory_system.encode_memory(
                            content,
                            memory_type,
                            context=context,
                            emotional_tags=emotion_tags,
                        )
                    )
                    if memory_id:
                        stored_ids.append(memory_id)
                        result["stored_id"] = memory_id
                        self._index_vector_memory(
                            memory_id,
                            content,
                            memory_type=memory_type,
                            context=context,
                            emotion_tags=emotion_tags,
                            inputs=inputs,
                        )
                if stored_ids:
                    result["stored_ids"] = stored_ids

            retrieve_payload = inputs.get("retrieve")
            if retrieve_payload is not None:
                query, memory_type, context = self._parse_retrieve_payload(retrieve_payload, inputs)
                if query is not None:
                    retrieved = self._run_async(
                        self.memory_system.retrieve_memory(
                            query,
                            memory_type=memory_type,
                            context=context,
                        )
                    )
                    result["retrieved"] = retrieved or []
                    vector_query_text, vector_matches = self._vector_similarity_search(
                        query,
                        memory_type=memory_type,
                        context=context,
                        inputs=inputs,
                    )

            result["statistics"] = self.memory_system.get_memory_statistics()
            if vector_matches:
                result["vector_matches"] = vector_matches
                result["statistics"]["vector_hits"] = len(vector_matches)
                if vector_query_text:
                    result["statistics"]["vector_query"] = vector_query_text

        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Memory process failure: %s", exc, exc_info=True)
            result.setdefault("error", str(exc))

        return result

    def _parse_store_payload(
        self,
        payload: Any,
        inputs: Dict[str, Any],
    ) -> Tuple[MemoryType, Any, Dict[str, Any], Dict[str, float]]:
        context = inputs.get("context", {}) or {}
        emotion_tags = inputs.get("emotion_tags", {}) or {}
        memory_type = self._resolve_memory_type(None)
        content: Any = None

        if isinstance(payload, dict):
            content = payload.get("content", payload.get("data"))
            memory_type = self._resolve_memory_type(payload.get("memory_type"))
            context = payload.get("context", context) or {}
            emotion_tags = payload.get("emotional_tags", payload.get("emotion", emotion_tags)) or {}
        else:
            content = payload

        if content is None:
            self.logger.debug("Skipping memory store request with empty content")
        return memory_type, content, context, emotion_tags

    def _parse_retrieve_payload(
        self,
        payload: Any,
        inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[MemoryType], Dict[str, Any]]:
        context = inputs.get("context", {}) or {}
        memory_type: Optional[MemoryType] = None

        if isinstance(payload, dict):
            memory_type = self._resolve_memory_type(payload.get("memory_type"))
            query = payload.get("query", payload)
            context = payload.get("context", context) or {}
        elif isinstance(payload, str):
            query = {"query": payload}
        else:
            query = {"query": str(payload)}

        return query, memory_type, context

    def _resolve_memory_type(self, value: Any) -> MemoryType:
        if isinstance(value, MemoryType):
            return value
        if isinstance(value, str):
            key = value.strip().upper()
            if key in MemoryType.__members__:
                return MemoryType[key]
        default_key = str(self.default_memory_type).strip().upper()
        return MemoryType.__members__.get(default_key, MemoryType.WORKING)

    def _run_async(self, coroutine):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result()

        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coroutine)
        finally:
            asyncio.set_event_loop(None)
            new_loop.close()

    def _index_vector_memory(
        self,
        memory_id: Optional[str],
        content: Any,
        *,
        memory_type: MemoryType,
        context: Dict[str, Any],
        emotion_tags: Dict[str, float],
        inputs: Dict[str, Any],
    ) -> None:
        if not getattr(self.vector_store, "is_available", False):
            return
        metadata = {
            "memory_type": memory_type.name if isinstance(memory_type, MemoryType) else str(memory_type),
            "timestamp": inputs.get("timestamp", time.time()),
            "source": inputs.get("source"),
        }
        metadata.update(self._flatten_metadata({"context": context, "emotion": emotion_tags}))
        try:
            self.vector_store.index_memory(memory_id, content, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug("Vector memory indexing failed: %s", exc)

    def _vector_similarity_search(
        self,
        query: Any,
        *,
        memory_type: Optional[MemoryType],
        context: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        if not getattr(self.vector_store, "is_available", False):
            return None, []

        query_text = self._extract_query_text(query)
        if not query_text:
            return None, []

        contextual_terms: List[str] = [query_text]
        for key, value in (context or {}).items():
            if isinstance(value, (str, int, float)):
                contextual_terms.append(f"{key}:{value}")
        search_text = " ".join(str(term) for term in contextual_terms if term is not None)

        top_k = inputs.get("vector_top_k")
        try:
            matches = self.vector_store.similarity_search(
                search_text,
                top_k=int(top_k) if top_k is not None else None,
                filters=None,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug("Vector search failed: %s", exc)
            matches = []

        if matches:
            for match in matches:
                match.setdefault("source", "vector_store")
                if memory_type is not None:
                    metadata = match.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata.setdefault(
                        "memory_type",
                        memory_type.name if isinstance(memory_type, MemoryType) else str(memory_type),
                    )
                    match["metadata"] = metadata
        return search_text, matches

    @staticmethod
    def _flatten_metadata(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in (data or {}).items():
            name = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(MemoryProcess._flatten_metadata(value, prefix=f"{name}_"))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[name] = value
            else:
                try:
                    flat[name] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    flat[name] = str(value)
        return flat

    @staticmethod
    def _extract_query_text(query: Any) -> Optional[str]:
        if isinstance(query, str):
            return query
        if isinstance(query, dict):
            for candidate in ("query", "text", "keywords", "content"):
                value = query.get(candidate)
                if isinstance(value, str) and value.strip():
                    return value
            collected = [str(value) for value in query.values() if isinstance(value, (str, int, float))]
            if collected:
                return " ".join(collected)
        if isinstance(query, (list, tuple)):
            collected = [str(value) for value in query if isinstance(value, (str, int, float))]
            if collected:
                return " ".join(collected)
        return None



def create_comprehensive_memory_system(config: Dict[str, Any]) -> ComprehensiveMemorySystem:
    """创建综合记忆系统"""
    return ComprehensiveMemorySystem(config)
