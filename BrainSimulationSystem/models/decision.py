"""
决策过程模块

实现基于输入和内部状态做出决策的功能。
"""

import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, Any, Deque
import numpy as np
import random

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess
from BrainSimulationSystem.models.action_selection_bg import ActionSelectionBG

try:  # pragma: no cover - optional dependency
    from BrainSimulationSystem.decision.deep_rl_agent import (
        DecisionRLAgent,
        RLAgentConfig,
        build_option_observation,
    )
except Exception:  # pragma: no cover - RL dependencies are optional
    DecisionRLAgent = None  # type: ignore[assignment]
    RLAgentConfig = None  # type: ignore[assignment]
    build_option_observation = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DecisionProcess(CognitiveProcess):
    """
    决策过程
    
    基于输入和内部状态做出决策
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化决策过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - decision_type: 决策类型
                - temperature: 决策温度（用于探索-利用权衡）
                - threshold: 决策阈值
        """
        super().__init__(network, params or {})

        # 决策历史
        self.decision_history = []

        # 奖励历史
        self.reward_history = []

        # 动作价值估计
        self.action_values = {}

        self._meta_adjustment_provider = None
        if isinstance(self.params, dict):
            provider = self.params.get("meta_decision_system") or self.params.get("meta_adjustment_provider")
            if provider is not None:
                self._meta_adjustment_provider = provider
            self.params.setdefault("exploration_rate", 0.1)

        # 强化学习代理（可选）
        self._rl_agent = None
        rl_params = self.params.get("rl", {}) if isinstance(self.params, dict) else {}
        if (
            rl_params.get("enabled")
            and DecisionRLAgent is not None
            and RLAgentConfig is not None
        ):
            config = RLAgentConfig(
                algorithm=rl_params.get("algorithm", "ppo"),
                policy=rl_params.get("policy", "MlpPolicy"),
                model_path=rl_params.get("model_path"),
                device=rl_params.get("device", "auto"),
                verbose=rl_params.get("verbose", 0),
            )
            try:
                self._rl_agent = DecisionRLAgent(config=config)
            except Exception:
                self._rl_agent = None
        self._rl_auto_update = bool(rl_params.get("auto_update", True))
        self._rl_update_interval = float(rl_params.get("update_interval_seconds", 30.0))
        self._rl_min_batch = int(rl_params.get("min_batch_size", 4))
        self._rl_train_steps = int(rl_params.get("train_timesteps", 256))
        self._rl_last_update = time.time() - self._rl_update_interval
        self._rl_update_thread: Optional[threading.Thread] = None
        self._rl_last_observation: Optional[np.ndarray] = None
        self._rl_last_context: Optional[str] = None
        self._rl_max_pending = int(rl_params.get("max_pending_outcomes", 64))
        self._rl_local_pending: Deque[Tuple[np.ndarray, float]] = deque(
            maxlen=self._rl_max_pending if self._rl_max_pending > 0 else None
        )

        bg_cfg: Dict[str, Any] = {}
        if isinstance(self.params, dict):
            candidate = self.params.get("basal_ganglia") or self.params.get("bg") or {}
            if isinstance(candidate, dict):
                bg_cfg = candidate
        self._bg_selector = ActionSelectionBG(bg_cfg)

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理决策
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - options: 决策选项
                - context: 决策上下文
                - reward: 上一次决策的奖励（可选）
                
        Returns:
            包含处理结果的字典
        """
        # 应用元决策调整（如有）
        self._apply_pending_meta_adjustments()

        # 获取参数
        decision_type = self.params.get("decision_type", "softmax")
        temperature = self.params.get("temperature", 1.0)
        threshold = self.params.get("threshold", 0.5)
        
        # 获取决策选项
        options = inputs.get("options", [])
        if not options:
            return {"decision": None, "confidence": 0.0}
        
        # 获取上下文
        context = inputs.get("context", {})
        try:
            temperature = self._emotion_modulated_temperature(float(temperature), context)
        except Exception:  # pragma: no cover - defensive fallback
            temperature = float(temperature)
        context_key = str(hash(str(context)))
        
        # 处理奖励（如果有）
        if "reward" in inputs:
            self._process_reward(inputs["reward"], context_key)
        else:
            # Clear stale RL observation alignment when no reward is supplied.
            self._rl_last_observation = None
            self._rl_last_context = None
        
        # 做出决策
        if decision_type == "softmax":
            # Softmax决策：基于价值的概率选择
            decision, confidence = self._softmax_decision(options, context_key, temperature)

        elif decision_type == "greedy":
            # 贪婪决策：选择价值最高的选项
            decision, confidence = self._greedy_decision(options, context_key)

        elif decision_type == "threshold":
            # 阈值决策：如果最高价值超过阈值则选择，否则随机
            decision, confidence = self._threshold_decision(options, context_key, threshold)

        elif decision_type == "rl":
            decision, confidence = self._rl_decision(options, context, temperature)
            if decision is None:
                decision, confidence = self._softmax_decision(options, context_key, temperature)

        elif decision_type in {"basal_ganglia", "bg"}:
            decision, confidence = self._basal_ganglia_decision(options, context_key, context)

        elif decision_type in {"planner", "plan"}:
            decision, confidence = self._planner_decision(options, context, context_key, temperature)

        elif decision_type in {"planner_bg", "plan_bg"}:
            decision, confidence = self._planner_basal_ganglia_decision(options, context, context_key)

        else:
            # 默认随机决策
            decision = random.choice(options)
            confidence = 1.0 / len(options)
        
        # 记录决策
        self.decision_history.append((decision, context_key))

        try:
            predicted_reward = float(self.action_values.get((context_key, decision), 0.0))
        except Exception:
            predicted_reward = 0.0

        return {
            "decision": decision,
            "confidence": confidence,
            "predicted_reward": predicted_reward,
            "action_values": {opt: self.action_values.get((context_key, opt), 0.0) for opt in options}
        }

    @staticmethod
    def _emotion_modulated_temperature(base: float, context: Any) -> float:
        """Modulate softmax temperature using emotion/limbic context (best effort)."""

        temperature = float(base)
        if temperature <= 0:
            temperature = 0.01

        if not isinstance(context, dict):
            return temperature

        scale = 1.0
        emotion_state = context.get("emotion_state")
        if isinstance(emotion_state, dict):
            try:
                valence = float(emotion_state.get("valence", 0.0))
            except Exception:
                valence = 0.0
            try:
                arousal = float(emotion_state.get("arousal", 0.0))
            except Exception:
                arousal = 0.0

            # Negative valence -> more conservative (lower temperature).
            if valence < -0.2:
                scale *= max(0.35, 1.0 + 0.8 * valence)
            # Positive valence -> slightly more exploratory (higher temperature).
            elif valence > 0.2:
                scale *= min(1.8, 1.0 + 0.4 * valence)

            # High arousal -> focus slightly (lower temperature), but keep effect mild.
            if arousal > 0.6:
                scale *= max(0.5, 1.0 - 0.25 * (arousal - 0.6))

        limbic = context.get("limbic")
        if isinstance(limbic, dict):
            bias = limbic.get("decision_bias")
            if isinstance(bias, dict) and bias.get("temperature_scale") is not None:
                try:
                    scale *= float(bias.get("temperature_scale", 1.0))
                except Exception:
                    pass

        return max(0.05, temperature * float(scale))

    def _basal_ganglia_decision(
        self,
        options: List[Any],
        context_key: str,
        context: Any,
    ) -> Tuple[Any, float]:
        values = [float(self.action_values.get((context_key, opt), 0.0)) for opt in options]
        selection = self._bg_selector.select(options, values, context=context if isinstance(context, dict) else None)
        return selection.action, float(selection.confidence)

    @staticmethod
    def _extract_plan_sequence(context: Any) -> List[Any]:
        if not isinstance(context, dict):
            return []
        for key in ("plan_sequence", "planner_sequence"):
            seq = context.get(key)
            if isinstance(seq, list):
                return seq
        planner = context.get("planner")
        if isinstance(planner, dict):
            seq = planner.get("sequence")
            if isinstance(seq, list):
                return seq
        return []

    def _planner_decision(
        self,
        options: List[Any],
        context: Any,
        context_key: str,
        temperature: float,
    ) -> Tuple[Any, float]:
        if not isinstance(context, dict):
            return self._softmax_decision(options, context_key, temperature)

        preferred = context.get("plan_next_action") or context.get("next_action")
        if preferred is not None and preferred in options:
            return preferred, 0.9

        sequence = self._extract_plan_sequence(context)
        if sequence:
            for step in sequence:
                if step in options:
                    return step, 0.75

        return self._softmax_decision(options, context_key, temperature)

    def _planner_basal_ganglia_decision(
        self,
        options: List[Any],
        context: Any,
        context_key: str,
    ) -> Tuple[Any, float]:
        if not isinstance(context, dict):
            return self._basal_ganglia_decision(options, context_key, context or {})

        preferred = context.get("plan_next_action") or context.get("next_action")
        candidate_pool: List[Any] = []
        if preferred is not None and preferred in options:
            candidate_pool.append(preferred)

        sequence = self._extract_plan_sequence(context)
        if sequence:
            for step in sequence:
                if step in options and step not in candidate_pool:
                    candidate_pool.append(step)
                if len(candidate_pool) >= 5:
                    break

        if not candidate_pool:
            candidate_pool = list(options)

        values = [float(self.action_values.get((context_key, opt), 0.0)) for opt in candidate_pool]
        selection = self._bg_selector.select(candidate_pool, values, context=context)
        return selection.action, float(selection.confidence)

    def set_meta_adjustment_provider(self, provider: Any) -> None:
        """设置元决策调整提供者"""
        self._meta_adjustment_provider = provider

    def _apply_pending_meta_adjustments(self) -> None:
        if not self._meta_adjustment_provider:
            return
        consumer = getattr(self._meta_adjustment_provider, "consume_pending_adjustments", None)
        if not callable(consumer):
            return
        adjustments = consumer('process')
        if not adjustments:
            return
        self._apply_process_adjustments(adjustments)

    def _apply_process_adjustments(self, adjustments: Dict[str, float]) -> None:
        if not isinstance(adjustments, dict):
            return

        learning_delta = adjustments.get('learning_rate')
        if learning_delta:
            current = float(self.params.get("learning_rate", 0.1))
            self.params["learning_rate"] = max(0.0, current + learning_delta)

        exploration_delta = adjustments.get('exploration_rate')
        if exploration_delta:
            current_exp = float(self.params.get("exploration_rate", 0.1))
            new_exp = max(0.0, min(1.0, current_exp + exploration_delta))
            self.params["exploration_rate"] = new_exp
            temp = float(self.params.get("temperature", 1.0))
            self.params["temperature"] = max(0.05, temp + exploration_delta)

    def _build_rl_observations(
        self,
        options: List[Any],
        context: Dict[str, Any],
    ) -> List[np.ndarray]:
        observations: List[np.ndarray] = []
        if build_option_observation is None:
            return observations

        context = context or {}
        total = len(options)
        for idx, opt in enumerate(options):
            if isinstance(opt, dict):
                observations.append(build_option_observation(context, opt, idx, total))
            else:
                observations.append(
                    build_option_observation(
                        context,
                        {"expected_value": float(idx == 0)},
                        idx,
                        total,
                    )
                )
        return observations

    def _rl_decision(
        self,
        options: List[Any],
        context: Dict[str, Any],
        temperature: float,
    ) -> Tuple[Optional[Any], float]:
        if not options or self._rl_agent is None:
            return None, 0.0

        observations = self._build_rl_observations(options, context)
        if not observations:
            return None, 0.0

        index, info = self._rl_agent.predict_action(observations)
        if index is None or not (0 <= index < len(options)):
            return None, 0.0

        decision = options[index]
        confidence = float(info.get("confidence", 0.0))
        try:
            self._rl_last_observation = observations[index]
            self._rl_last_context = str(hash(str(context)))
        except Exception:
            self._rl_last_observation = None
            self._rl_last_context = None

        if confidence == 0.0 and info.get("scores"):
            scores = info["scores"]
            exp_values = np.exp(np.array(scores) / max(temperature, 1e-3))
            probabilities = exp_values / np.sum(exp_values)
            confidence = float(probabilities[index])

        return decision, confidence
    
    def _process_reward(self, reward: float, context_key: str) -> None:
        """
        处理奖励
        
        Args:
            reward: 奖励值
            context_key: 上下文键
        """
        # 记录奖励
        self.reward_history.append(reward)
        
        # 如果有决策历史，更新最近决策的价值
        if self.decision_history:
            last_decision, last_context = self.decision_history[-1]
            
            # 只有在相同上下文下才更新
            if last_context == context_key:
                # 获取当前价值估计
                key = (last_context, last_decision)
                current_value = self.action_values.get(key, 0.0)
                
                # 学习率
                alpha = self.params.get("learning_rate", 0.1)
                 
                # 更新价值估计：Q(s,a) = Q(s,a) + α[r - Q(s,a)]
                new_value = current_value + alpha * (reward - current_value)
                self.action_values[key] = new_value
                if (
                    self._rl_agent is not None
                    and self._rl_last_observation is not None
                    and last_context == self._rl_last_context
                ):
                    self._queue_rl_outcome(self._rl_last_observation, float(reward))
                    self._rl_last_observation = None
                    self._rl_last_context = None
            else:
                # 上下文不匹配，丢弃旧的 RL 观测
                self._rl_last_observation = None
                self._rl_last_context = None

    # ------------------------------------------------------------------ #
    def _queue_rl_outcome(self, observation: np.ndarray, reward: float) -> None:
        """Record an outcome for online RL updates, with lightweight capping."""

        if self._rl_agent is None:
            return

        try:
            if self._rl_max_pending > 0 and len(self._rl_local_pending) >= self._rl_max_pending:
                self._rl_local_pending.popleft()
            self._rl_local_pending.append((observation, reward))
            self._rl_agent.record_outcome(observation, reward)
            pending = getattr(self._rl_agent, "_pending_observations", None)
            if (
                self._rl_max_pending > 0
                and isinstance(pending, list)
                and len(pending) > self._rl_max_pending
            ):
                del pending[: len(pending) - self._rl_max_pending]
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to queue RL outcome for online update", exc_info=True)
            return

        self._maybe_schedule_rl_update()

    def _pending_outcomes(self) -> int:
        if self._rl_agent is None:
            return 0
        try:
            pending = getattr(self._rl_agent, "_pending_observations", [])
            return len(pending)
        except Exception:
            return len(self._rl_local_pending)

    def _maybe_schedule_rl_update(self) -> None:
        if not self._rl_auto_update or self._rl_agent is None:
            return
        if self._rl_update_thread and self._rl_update_thread.is_alive():
            return
        if self._pending_outcomes() < self._rl_min_batch:
            return
        if (time.time() - self._rl_last_update) < self._rl_update_interval:
            return

        self._rl_update_thread = threading.Thread(
            target=self._run_rl_update, name="decision-rl-update", daemon=True
        )
        self._rl_update_thread.start()

    def _run_rl_update(self) -> None:
        agent = self._rl_agent
        if agent is None:
            return
        started = time.time()
        try:
            result = agent.update(total_timesteps=self._rl_train_steps)
            logger.debug("Decision RL online update result: %s", result)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Decision RL online update failed", exc_info=True)
        finally:
            self._rl_last_update = started
    
    def _softmax_decision(
        self, 
        options: List[Any], 
        context_key: str, 
        temperature: float
    ) -> Tuple[Any, float]:
        """
        Softmax决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            temperature: 温度参数
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 应用softmax
        if temperature <= 0:
            temperature = 0.01  # 避免除以零
        
        # 减去最大值以提高数值稳定性
        max_value = max(values)
        exp_values = [np.exp((v - max_value) / temperature) for v in values]
        sum_exp = sum(exp_values)
        
        if sum_exp == 0:
            # 如果所有指数值都是0，使用均匀分布
            probabilities = [1.0 / len(options) for _ in options]
        else:
            probabilities = [ev / sum_exp for ev in exp_values]
        
        # 根据概率选择
        choice_idx = random.choices(range(len(options)), weights=probabilities)[0]
        choice = options[choice_idx]
        confidence = probabilities[choice_idx]
        
        return choice, confidence
    
    def _greedy_decision(
        self, 
        options: List[Any], 
        context_key: str
    ) -> Tuple[Any, float]:
        """
        贪婪决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 找到最大价值的索引
        max_value = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_value]
        
        # 如果有多个最大值，随机选择一个
        choice_idx = random.choice(max_indices)
        choice = options[choice_idx]
        
        # 计算置信度：最大值与平均值的差距
        avg_value = sum(values) / len(values)
        if avg_value == max_value:
            confidence = 1.0 / len(options)
        else:
            # 归一化置信度到[0,1]
            confidence = min(1.0, max(0.0, (max_value - avg_value) / max_value))
        
        return choice, confidence
    
    def _threshold_decision(
        self, 
        options: List[Any], 
        context_key: str, 
        threshold: float
    ) -> Tuple[Any, float]:
        """
        阈值决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            threshold: 决策阈值
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 找到最大价值的索引
        max_value = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_value]
        
        # 如果最大值超过阈值，选择它
        if max_value >= threshold:
            choice_idx = random.choice(max_indices)
            choice = options[choice_idx]
            confidence = max_value
        else:
            # 否则随机选择
            choice_idx = random.randrange(len(options))
            choice = options[choice_idx]
            confidence = max_value / threshold  # 归一化置信度
        
        return choice, confidence
