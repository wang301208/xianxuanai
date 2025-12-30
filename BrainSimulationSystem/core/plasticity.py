"""
突触可塑性模型，包括短时可塑性（STP）和长时可塑性（LTP/LTD）。
Synaptic plasticity models, including Short-Term Plasticity (STP) and Long-Term Plasticity (LTP/LTD).
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple

class ShortTermPlasticity:
    """
    短时可塑性（STP）模型，基于Tsodyks-Markram (TM)模型。
    模拟突触的易化（facilitation）和抑制（depression）。
    """
    
    def __init__(self, tau_rec: float = 800.0, tau_fac: float = 50.0, U: float = 0.5):
        """
        初始化短时可塑性模型。

        Args:
            tau_rec (float): 抑制的恢复时间常数 (ms)。
            tau_fac (float): 易化的恢复时间常数 (ms)。
            U (float): 基础资源利用率（即单个脉冲的释放概率）。
        """
        self.tau_rec = float(tau_rec)
        self.tau_fac = float(tau_fac)
        self.U = float(U)

        # Keep an immutable baseline so neuromodulator effects do not accumulate.
        self._baseline = {
            "tau_rec": float(self.tau_rec),
            "tau_fac": float(self.tau_fac),
            "U": float(self.U),
        }
        
        # 状态变量
        self.u = U      # 当前利用率 (facilitation)
        self.x = 1.0    # 可用资源分数 (depression)
        
        self.logger = logging.getLogger("STP")

    def apply_neuromodulation(self, neuromodulators: Optional[Dict[str, float]]) -> None:
        """Apply neuromodulator-dependent parameter changes (non-accumulating).

        This keeps ``tau_rec/tau_fac/U`` stable across steps by recomputing
        effective values from a stored baseline each call.
        """
        baseline = self._baseline
        self.tau_rec = float(baseline["tau_rec"])
        self.tau_fac = float(baseline["tau_fac"])
        self.U = float(baseline["U"])

        if not neuromodulators:
            return

        def _level(name: str) -> float:
            try:
                return float(neuromodulators.get(name, 0.0))
            except Exception:
                return 0.0

        dopamine = _level("dopamine")
        acetylcholine = _level("acetylcholine")
        norepinephrine = _level("norepinephrine")
        serotonin = _level("serotonin")

        # Heuristic mapping (bounded): DA tends to increase release probability and stabilise
        # synapses during reward; ACh increases responsiveness; 5-HT can dampen release.
        self.U = float(np.clip(self.U * (1.0 + 0.6 * dopamine + 0.2 * acetylcholine - 0.2 * serotonin), 0.01, 0.95))
        self.tau_rec = float(np.clip(self.tau_rec * (1.0 - 0.25 * acetylcholine + 0.15 * serotonin), 10.0, 5000.0))
        self.tau_fac = float(np.clip(self.tau_fac * (1.0 - 0.15 * norepinephrine + 0.1 * serotonin), 5.0, 1000.0))
    
    def process_spike(self) -> float:
        """
        处理一次突触前脉冲，计算释放概率并更新状态变量。

        Returns:
            float: 本次脉冲的有效释放概率 (u * x)。
        """
        # 计算本次脉冲的释放概率
        release_probability = self.u * self.x
        
        # 更新状态变量
        # 易化：u在脉冲后增加
        self.u += self.U * (1 - self.u)
        # 抑制：x在脉冲后减少
        self.x -= release_probability
        
        return release_probability
    

    def update(self, dt: float):
        """
        在每个时间步更新状态变量，模拟其向基准值的恢复过程。

        Args:
            dt (float): 时间步长 (ms)。
        """
        # 恢复可用资源 (x -> 1)
        self.x += (1 - self.x) / self.tau_rec * dt
        
        # 恢复利用率 (u -> U)
        self.u += (self.U - self.u) / self.tau_fac * dt
        
        # 确保变量在合理范围内
        self.x = np.clip(self.x, 0.0, 1.0)
        self.u = np.clip(self.u, self.U, 1.0)


class LongTermPlasticity:
    """
    长时可塑性模型，基于脉冲时间依赖可塑性（STDP）。
    """
    
    def __init__(self, learning_rate: float = 0.01, metaplasticity: bool = True):
        """
        初始化长时可塑性模型。

        Args:
            learning_rate (float): 学习速率，调节权重变化的幅度。
            metaplasticity (bool): 是否启用元可塑性（滑动阈值）。
        """
        self.learning_rate = float(learning_rate)
        self.metaplasticity = metaplasticity

        self._baseline_learning_rate = float(self.learning_rate)
        
        # STDP窗口参数
        self.tau_plus = 20.0        # LTP时间窗口 (ms)
        self.tau_minus = 20.0       # LTD时间窗口 (ms)
        self.A_plus = 0.01          # LTP幅度
        self.A_minus = 0.0105       # LTD幅度 (略大于LTP以保持稳定)
        
        # 迹变量 (traces)
        self.pre_trace = 0.0
        self.post_trace = 0.0
        
        # 元可塑性相关变量
        self.theta = 0.0            # 动态修改阈值
        self.tau_theta = 10000.0    # 阈值更新的时间常数 (ms)
        self.weight_changes = []
        
        self.logger = logging.getLogger("LTP")
    
    def process_pre_spike(self, current_weight: float) -> float:
        """
        处理突触前脉冲事件。
        1. 增加突触前迹。
        2. 根据当前的突触后迹计算LTD。

        Returns:
            float: 计算出的权重变化量 (Δw)。
        """
        weight_change = 0.0
        # LTD：突触前发放时检查突触后迹
        if self.post_trace > 0:
            weight_change = -self.A_minus * self.post_trace
            if self.metaplasticity:
                weight_change *= self._metaplasticity_factor(current_weight)
            self.weight_changes.append(weight_change)

        # 更新突触前迹
        self.pre_trace += 1.0
        
        return weight_change * self.learning_rate
    
    def process_post_spike(self, current_weight: float) -> float:
        """
        处理突触后脉冲事件。
        1. 增加突触后迹。
        2. 根据当前的突触前迹计算LTP。

        Returns:
            float: 计算出的权重变化量 (Δw)。
        """
        weight_change = 0.0
        # LTP：突触后发放时检查突触前迹
        if self.pre_trace > 0:
            weight_change = self.A_plus * self.pre_trace
            if self.metaplasticity:
                weight_change *= self._metaplasticity_factor(current_weight)
            self.weight_changes.append(weight_change)

        # 更新突触后迹
        self.post_trace += 1.0
        
        return weight_change * self.learning_rate

    def apply_neuromodulation(self, neuromodulators: Optional[Dict[str, float]]) -> None:
        """Modulate plasticity gains using neuromodulator levels (non-accumulating)."""
        self.learning_rate = float(self._baseline_learning_rate)
        if not neuromodulators:
            return

        def _level(name: str) -> float:
            try:
                return float(neuromodulators.get(name, 0.0))
            except Exception:
                return 0.0

        dopamine = _level("dopamine")
        acetylcholine = _level("acetylcholine")
        norepinephrine = _level("norepinephrine")
        serotonin = _level("serotonin")

        # 3-factor style: treat DA as a gating signal on effective learning rate.
        # ACh/NE increase learning readiness; 5-HT can bias towards depression/slow learning.
        gate = 1.0 + 1.2 * dopamine + 0.3 * acetylcholine + 0.2 * norepinephrine - 0.4 * serotonin
        self.learning_rate = float(np.clip(self.learning_rate * gate, 0.0, 1.0))

    def _metaplasticity_factor(self, current_weight: float) -> float:
        """计算元可塑性因子"""
        # BCM-like规则：基于突触后活动历史调节
        recent_activity = len([c for c in self.weight_changes[-100:] if c > 0])
        activity_factor = recent_activity / 100.0
        
        # 滑动阈值
        self.theta = 0.5 * activity_factor
        
        # 权重依赖的调节
        if current_weight > self.theta:
            return 1.0 + 0.5 * (current_weight - self.theta)
        else:
            return 1.0 - 0.3 * (self.theta - current_weight)

    def update(self, dt: float):
        """
        在每个时间步更新迹变量，使其指数衰减。

        Args:
            dt (float): 时间步长 (ms)。
        """
        # 迹的指数衰减
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)

        # 阈值更新
        if self.metaplasticity:
            recent_ltp = len([c for c in self.weight_changes[-1000:] if c > 0])
            target_theta = recent_ltp / 1000.0
            self.theta += (target_theta - self.theta) / self.tau_theta * dt
