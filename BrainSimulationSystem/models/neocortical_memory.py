"""
新皮层记忆系统实现
Neocortical Memory System Implementation

实现新皮层的语义记忆和程序性记忆机制
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
import json
import hashlib
import types

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency / test stubs
    nx = None  # type: ignore[assignment]


if nx is None or not hasattr(nx, "Graph"):
    class _FallbackGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []

        def add_node(self, node, **attrs):
            self._nodes[node] = dict(attrs or {})

        def add_edge(self, u, v, **attrs):
            self._edges.append((u, v, dict(attrs or {})))

        def nodes(self, data: bool = False):
            if data:
                return list(self._nodes.items())
            return list(self._nodes.keys())

    try:
        if nx is None:
            nx = types.SimpleNamespace(Graph=_FallbackGraph)  # type: ignore[assignment]
        else:
            nx.Graph = _FallbackGraph  # type: ignore[attr-defined]
    except Exception:
        nx = types.SimpleNamespace(Graph=_FallbackGraph)  # type: ignore[assignment]
from scipy.spatial.distance import cosine

class NeocorticalMemorySystem:
    """新皮层记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 语义记忆网络
        self.semantic_network = nx.Graph()
        self.concept_embeddings = {}

        # Optional symbolic semantic network (e.g. language SemanticNetwork) for
        # concept-level semantic memory interop.
        self.symbolic_network: Optional[Any] = None

        # Capacity + forgetting configuration (lightweight defaults; opt-in).
        self.max_concepts = int(self.config.get("max_concepts", 50_000))
        forgetting_cfg = self.config.get("forgetting", {})
        if not isinstance(forgetting_cfg, dict):
            forgetting_cfg = {}
        self.forgetting_enabled = bool(forgetting_cfg.get("enabled", False))
        self.strength_decay_rate = float(forgetting_cfg.get("strength_decay_rate", 0.0002))
        self.edge_decay_rate = float(forgetting_cfg.get("edge_decay_rate", 0.0001))
        self.min_concept_strength = float(forgetting_cfg.get("min_concept_strength", 0.05))
        self.min_edge_weight = float(forgetting_cfg.get("min_edge_weight", 0.05))
        self.prune_batch = int(forgetting_cfg.get("prune_batch", 256))

        self.embedding_size = int(self.config.get("embedding_size", 300))
        self.embedding_update_rate = float(self.config.get("embedding_update_rate", 0.2))
        self.similarity_threshold = float(self.config.get("similarity_threshold", 0.3))
        
        # 程序性记忆
        self.procedural_memories = {}
        self.skill_hierarchies = {}
        
        # 分布式表征
        self.cortical_areas = {
            'temporal': {},    # 语义记忆
            'frontal': {},     # 程序性记忆
            'parietal': {},    # 空间记忆
            'occipital': {}    # 视觉记忆
        }
        
        self.logger = logging.getLogger("NeocorticalMemory")
        self.stable_concepts = bool(self.config.get("stable_concepts", True))
        if self.stable_concepts:
            self.encode_semantic_memory = self._encode_semantic_memory_stable  # type: ignore[assignment]
            self.retrieve_semantic_memory = self._retrieve_semantic_memory_stable  # type: ignore[assignment]

    @staticmethod
    def _normalise_concept(concept: Any) -> str:
        value = str(concept or "").strip()
        if not value:
            return "unknown"
        return value.lower()

    @staticmethod
    def _concept_id(concept_key: str) -> str:
        return f"concept::{concept_key}"

    def attach_symbolic_network(self, semantic_network: Any) -> None:
        """Attach an external symbolic semantic network to mirror concepts/relations."""

        self.symbolic_network = semantic_network

    def _touch_concept(self, concept_id: str, *, boost: float = 0.02) -> None:
        entry = self.cortical_areas.get("temporal", {}).get(concept_id)
        if not isinstance(entry, dict):
            return
        try:
            entry["strength"] = float(np.clip(float(entry.get("strength", 1.0)) + float(boost), 0.0, 2.0))
        except Exception:
            entry["strength"] = 1.0
        entry["last_access_time"] = time.time()

    def _ensure_symbolic_node(self, concept: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        network = self.symbolic_network
        if network is None:
            return
        add_node = getattr(network, "add_node", None)
        if callable(add_node):
            try:
                add_node(str(concept), attributes or {})
            except Exception:
                return

    def _ensure_symbolic_relation(self, head: str, dependent: str, relation_type: str, strength: float) -> None:
        network = self.symbolic_network
        if network is None:
            return
        add_relation = getattr(network, "add_relation", None)
        if callable(add_relation):
            try:
                add_relation(str(head), str(dependent), str(relation_type), strength=float(strength))
            except Exception:
                return
    
    def _generate_concept_embedding_stable(self, concept: str, properties: Dict[str, Any]) -> np.ndarray:
        embedding_size = int(self.embedding_size)
        if embedding_size <= 0:
            embedding_size = 300

        payload = json.dumps({"concept": str(concept), "properties": properties or {}}, sort_keys=True, default=str)
        seed = int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        embedding = rng.uniform(-0.1, 0.1, embedding_size).astype(float)

        concept_hash = int(hashlib.md5(str(concept).encode("utf-8")).hexdigest()[:8], 16) % embedding_size
        embedding[concept_hash] += 0.5

        for prop, value in (properties or {}).items():
            tag = f"{prop}_{value}"
            prop_hash = int(hashlib.md5(tag.encode("utf-8")).hexdigest()[:8], 16) % embedding_size
            embedding[prop_hash] += 0.3

        denom = float(np.linalg.norm(embedding)) + 1e-12
        return embedding / denom

    def _encode_semantic_memory_stable(
        self,
        concept: str,
        properties: Dict[str, Any],
        associations: Optional[List[str]] = None,
    ) -> str:
        concept_key = self._normalise_concept(concept)
        concept_id = self._concept_id(concept_key)
        props = dict(properties or {})

        embedding = self._generate_concept_embedding_stable(concept_key, props)
        if concept_id in self.concept_embeddings:
            try:
                prev = np.asarray(self.concept_embeddings[concept_id], dtype=float).reshape(-1)
                lr = float(np.clip(self.embedding_update_rate, 0.0, 1.0))
                merged = (1.0 - lr) * prev + lr * embedding
                denom = float(np.linalg.norm(merged)) + 1e-12
                embedding = merged / denom
            except Exception:
                pass
        self.concept_embeddings[concept_id] = embedding

        if concept_id not in self.semantic_network:
            self.semantic_network.add_node(
                concept_id,
                concept=str(concept),
                properties=dict(props),
                embedding=embedding,
            )
        else:
            try:
                node_data = self.semantic_network.nodes[concept_id]
                node_data["concept"] = str(concept)
                node_props = node_data.get("properties")
                if isinstance(node_props, dict):
                    node_props.update(props)
                else:
                    node_data["properties"] = dict(props)
                node_data["embedding"] = embedding
            except Exception:
                pass

        temporal = self.cortical_areas.setdefault("temporal", {})
        entry = temporal.get(concept_id)
        if not isinstance(entry, dict):
            temporal[concept_id] = {
                "concept": str(concept),
                "properties": dict(props),
                "encoding_time": time.time(),
                "last_access_time": time.time(),
                "strength": 1.0,
            }
        else:
            entry["concept"] = str(concept)
            if isinstance(entry.get("properties"), dict):
                entry["properties"].update(props)
            else:
                entry["properties"] = dict(props)
            entry["last_access_time"] = time.time()
            try:
                entry["strength"] = float(np.clip(float(entry.get("strength", 1.0)) + 0.05, 0.0, 2.0))
            except Exception:
                entry["strength"] = 1.0

        self._ensure_symbolic_node(concept_key, {"source": "semantic_memory"})

        assoc_list = list(associations or [])
        for assoc_concept in assoc_list:
            assoc_key = self._normalise_concept(assoc_concept)
            if assoc_key == concept_key:
                continue
            assoc_id = self._concept_id(assoc_key)

            assoc_embedding = self.concept_embeddings.get(assoc_id)
            if assoc_embedding is None:
                assoc_embedding = self._generate_concept_embedding_stable(assoc_key, {})
                self.concept_embeddings[assoc_id] = assoc_embedding
                self.semantic_network.add_node(
                    assoc_id,
                    concept=str(assoc_concept),
                    properties={},
                    embedding=assoc_embedding,
                )
                temporal.setdefault(
                    assoc_id,
                    {
                        "concept": str(assoc_concept),
                        "properties": {},
                        "encoding_time": time.time(),
                        "last_access_time": time.time(),
                        "strength": 0.6,
                    },
                )

            try:
                similarity = float(np.dot(embedding, np.asarray(assoc_embedding, dtype=float).reshape(-1)))
            except Exception:
                similarity = 0.0
            if similarity > float(self.similarity_threshold):
                try:
                    if self.semantic_network.has_edge(concept_id, assoc_id):
                        current_weight = self.semantic_network[concept_id][assoc_id].get("weight", 0.0)
                        self.semantic_network[concept_id][assoc_id]["weight"] = max(float(current_weight), float(similarity))
                    else:
                        self.semantic_network.add_edge(concept_id, assoc_id, weight=float(similarity))
                except Exception:
                    pass

            self._ensure_symbolic_node(assoc_key, {"source": "semantic_memory"})
            self._ensure_symbolic_relation(concept_key, assoc_key, "semantic_association", float(similarity))
            self._ensure_symbolic_relation(assoc_key, concept_key, "semantic_association", float(similarity))

        self._prune_to_capacity()
        return concept_id

    def _retrieve_semantic_memory_stable(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_key = self._normalise_concept(query)
        query_embedding = self._generate_concept_embedding_stable(query_key, context or {})

        similarities = []
        for concept_id, embedding in self.concept_embeddings.items():
            try:
                similarity = float(np.dot(query_embedding, np.asarray(embedding, dtype=float).reshape(-1)))
            except Exception:
                similarity = 1.0 - cosine(query_embedding, embedding)
            similarities.append((concept_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for concept_id, similarity in similarities[:10]:
            if similarity <= 0.5:
                continue
            try:
                node_data = self.semantic_network.nodes[concept_id]
            except Exception:
                continue
            self._touch_concept(concept_id, boost=0.02)
            strength = None
            try:
                strength = float(self.cortical_areas.get("temporal", {}).get(concept_id, {}).get("strength", 1.0))
            except Exception:
                strength = None
            results.append(
                {
                    "concept_id": concept_id,
                    "concept": node_data.get("concept"),
                    "properties": node_data.get("properties"),
                    "similarity": float(similarity),
                    "strength": strength,
                }
            )
        return results

    def update(self, dt: float, *, rehearsal_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.forgetting_enabled and self.max_concepts <= 0:
            return {}

        dt_value = float(dt) if isinstance(dt, (int, float, np.floating)) else 0.0
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            dt_value = 0.0

        rehearsal = {self._concept_id(self._normalise_concept(c)) for c in (rehearsal_concepts or [])}

        pruned_concepts = 0
        pruned_edges = 0

        if self.forgetting_enabled and dt_value > 0.0:
            temporal = self.cortical_areas.get("temporal", {})
            if isinstance(temporal, dict):
                decay = float(self.strength_decay_rate)
                for concept_id, entry in list(temporal.items()):
                    if concept_id in rehearsal:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    try:
                        strength = float(entry.get("strength", 1.0))
                    except Exception:
                        strength = 1.0
                    strength *= float(np.exp(-decay * dt_value))
                    entry["strength"] = float(strength)

                to_drop = []
                for concept_id, entry in temporal.items():
                    if concept_id in rehearsal:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    try:
                        strength_value = float(entry.get("strength", 1.0))
                    except Exception:
                        strength_value = 1.0
                    if strength_value < float(self.min_concept_strength):
                        to_drop.append(concept_id)

                for concept_id in to_drop[: max(0, int(self.prune_batch))]:
                    temporal.pop(concept_id, None)
                    self.concept_embeddings.pop(concept_id, None)
                    try:
                        if concept_id in self.semantic_network:
                            self.semantic_network.remove_node(concept_id)
                    except Exception:
                        pass
                    pruned_concepts += 1

            try:
                edge_decay = float(self.edge_decay_rate)
                if edge_decay > 0.0:
                    for u, v, data in list(self.semantic_network.edges(data=True)):
                        try:
                            weight = float(data.get("weight", 0.0))
                        except Exception:
                            weight = 0.0
                        weight *= float(np.exp(-edge_decay * dt_value))
                        data["weight"] = float(weight)

                to_remove = []
                for u, v, data in self.semantic_network.edges(data=True):
                    try:
                        weight = float(data.get("weight", 0.0))
                    except Exception:
                        weight = 0.0
                    if weight < float(self.min_edge_weight):
                        to_remove.append((u, v))
                for u, v in to_remove[: max(0, int(self.prune_batch))]:
                    try:
                        self.semantic_network.remove_edge(u, v)
                        pruned_edges += 1
                    except Exception:
                        continue
            except Exception:
                pass

            symbolic = self.symbolic_network
            pruner = getattr(symbolic, "prune_relations", None) if symbolic is not None else None
            if callable(pruner):
                try:
                    pruner(min_strength=float(self.min_edge_weight), drop_isolated=False)
                except Exception:
                    pass

        self._prune_to_capacity()
        return {"pruned_concepts": int(pruned_concepts), "pruned_edges": int(pruned_edges)}

    def _prune_to_capacity(self) -> None:
        if self.max_concepts <= 0:
            return
        try:
            max_allowed = int(self.max_concepts)
        except Exception:
            return
        if max_allowed <= 0:
            return

        temporal = self.cortical_areas.get("temporal", {})
        if not isinstance(temporal, dict):
            return
        if len(temporal) <= max_allowed:
            return

        scored = []
        for concept_id, entry in temporal.items():
            if not isinstance(entry, dict):
                continue
            try:
                strength = float(entry.get("strength", 1.0))
            except Exception:
                strength = 1.0
            last_access = entry.get("last_access_time", entry.get("encoding_time", 0.0))
            try:
                last_access_value = float(last_access)
            except Exception:
                last_access_value = 0.0
            scored.append((strength, last_access_value, concept_id))

        scored.sort(key=lambda t: (t[0], t[1]))
        overflow = max(0, len(temporal) - max_allowed)
        for _, __, concept_id in scored[:overflow]:
            temporal.pop(concept_id, None)
            self.concept_embeddings.pop(concept_id, None)
            try:
                if concept_id in self.semantic_network:
                    self.semantic_network.remove_node(concept_id)
            except Exception:
                pass

    # Stable semantic memory helpers (used when stable_concepts=True).
    def encode_semantic_memory(self, concept: str, properties: Dict[str, Any],
                             associations: List[str] = None) -> str:
        """编码语义记忆"""
        
        concept_id = f"concept_{concept}_{int(time.time())}"
        
        # 生成概念嵌入
        embedding = self._generate_concept_embedding(concept, properties)
        self.concept_embeddings[concept_id] = embedding
        
        # 添加到语义网络
        self.semantic_network.add_node(concept_id, 
                                     concept=concept, 
                                     properties=properties,
                                     embedding=embedding)
        
        # 建立关联
        if associations:
            for assoc_concept in associations:
                # 找到相关概念
                related_nodes = [
                    node for node, data in self.semantic_network.nodes(data=True)
                    if data.get('concept') == assoc_concept
                ]
                
                for related_node in related_nodes:
                    # 计算语义相似性
                    similarity = self._calculate_semantic_similarity(
                        concept_id, related_node
                    )
                    
                    if similarity > 0.3:
                        self.semantic_network.add_edge(
                            concept_id, related_node, weight=similarity
                        )
        
        # 存储到皮层区域
        self.cortical_areas['temporal'][concept_id] = {
            'concept': concept,
            'properties': properties,
            'encoding_time': time.time(),
            'strength': 1.0
        }
        
        self.logger.debug(f"Encoded semantic memory: {concept}")
        return concept_id
    
    def _generate_concept_embedding(self, concept: str, 
                                  properties: Dict[str, Any]) -> np.ndarray:
        """生成概念嵌入"""
        
        # 简化的嵌入生成
        embedding_size = 300
        embedding = np.random.uniform(-0.1, 0.1, embedding_size)
        
        # 基于概念名称
        concept_hash = hash(concept) % embedding_size
        embedding[concept_hash] += 0.5
        
        # 基于属性
        for prop, value in properties.items():
            prop_hash = hash(f"{prop}_{value}") % embedding_size
            embedding[prop_hash] += 0.3
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _calculate_semantic_similarity(self, concept1_id: str, 
                                     concept2_id: str) -> float:
        """计算语义相似性"""
        
        if (concept1_id not in self.concept_embeddings or 
            concept2_id not in self.concept_embeddings):
            return 0.0
        
        emb1 = self.concept_embeddings[concept1_id]
        emb2 = self.concept_embeddings[concept2_id]
        
        # 余弦相似性
        similarity = 1.0 - cosine(emb1, emb2)
        return max(0.0, similarity)
    
    def retrieve_semantic_memory(self, query: str, 
                               context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """检索语义记忆"""
        
        # 生成查询嵌入
        query_embedding = self._generate_concept_embedding(query, context or {})
        
        # 计算与所有概念的相似性
        similarities = []
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = 1.0 - cosine(query_embedding, embedding)
            similarities.append((concept_id, similarity))
        
        # 排序并返回最相似的
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for concept_id, similarity in similarities[:10]:  # 返回前10个
            if similarity > 0.5:  # 相似性阈值
                node_data = self.semantic_network.nodes[concept_id]
                results.append({
                    'concept_id': concept_id,
                    'concept': node_data['concept'],
                    'properties': node_data['properties'],
                    'similarity': similarity
                })
        
        return results
    
    def encode_procedural_memory(self, skill_name: str, 
                               procedure: List[Dict[str, Any]]) -> str:
        """编码程序性记忆"""
        
        skill_id = f"skill_{skill_name}_{int(time.time())}"
        
        # 存储程序
        self.procedural_memories[skill_id] = {
            'skill_name': skill_name,
            'procedure': procedure,
            'proficiency': 0.1,  # 初始熟练度
            'practice_count': 0,
            'encoding_time': time.time()
        }
        
        # 构建技能层次
        self._build_skill_hierarchy(skill_id, skill_name, procedure)
        
        # 存储到额叶区域
        self.cortical_areas['frontal'][skill_id] = self.procedural_memories[skill_id]
        
        self.logger.debug(f"Encoded procedural memory: {skill_name}")
        return skill_id
    
    def _build_skill_hierarchy(self, skill_id: str, skill_name: str,
                             procedure: List[Dict[str, Any]]):
        """构建技能层次"""
        
        # 分解为子技能
        sub_skills = []
        for step in procedure:
            if 'sub_skill' in step:
                sub_skills.append(step['sub_skill'])
        
        self.skill_hierarchies[skill_id] = {
            'skill_name': skill_name,
            'sub_skills': sub_skills,
            'complexity': len(procedure),
            'dependencies': []
        }
    
    def practice_skill(self, skill_id: str, performance_feedback: float) -> Dict[str, Any]:
        """练习技能"""
        
        if skill_id not in self.procedural_memories:
            return {'error': 'Skill not found'}
        
        skill = self.procedural_memories[skill_id]
        
        # 更新练习次数
        skill['practice_count'] += 1
        
        # 更新熟练度（基于反馈）
        learning_rate = 0.1 / (1 + skill['practice_count'] * 0.01)  # 递减学习率
        proficiency_change = learning_rate * (performance_feedback - skill['proficiency'])
        skill['proficiency'] += proficiency_change
        skill['proficiency'] = np.clip(skill['proficiency'], 0.0, 1.0)
        
        return {
            'skill_id': skill_id,
            'new_proficiency': skill['proficiency'],
            'practice_count': skill['practice_count'],
            'proficiency_change': proficiency_change
        }
