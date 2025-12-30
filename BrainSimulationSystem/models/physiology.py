"""
生理模型模块
Physiology Models Module

包含了大脑模拟中使用的生理过程模型。
- BloodFlowModel: 血流动力学模型
- MetabolismModel: 代谢模型
"""
from typing import Dict

import numpy as np

class BloodFlowModel:
    """血流动力学模型"""
    
    def __init__(self, tissue_volume: float):
        self.tissue_volume = tissue_volume
        
        # 血流参数
        self.baseline_flow = 50.0  # ml/100g/min
        self.current_flow = self.baseline_flow
        self.oxygen_extraction = 0.4
        self.glucose_consumption = 5.0  # mg/100g/min
        
        # 血管反应性
        self.vascular_reactivity = 1.0
        self.autoregulation_strength = 0.8
        
        # BOLD信号相关
        self.deoxyhemoglobin = 0.6
        self.blood_volume = 0.04  # fraction
        
    def update(
        self,
        dt: float,
        neural_activity: float,
        *,
        metabolic_demand: float | None = None,
        vasoactive_signal: float | None = None,
    ) -> Dict[str, float]:
        """更新血流动力学（神经活动-血流-代谢粗粒度耦合）

        参数:
        - dt: 毫秒
        - neural_activity: 归一化活动水平（0..1）
        - metabolic_demand: 可选的代谢需求（0..1），默认等于 neural_activity
        - vasoactive_signal: 可选的胶质/神经调质血管驱动（0..1），默认等于 metabolic_demand
        """
        
        try:
            activity = float(neural_activity)
        except Exception:
            activity = 0.0

        try:
            demand = float(activity if metabolic_demand is None else metabolic_demand)
        except Exception:
            demand = float(activity)

        try:
            vaso = float(demand if vasoactive_signal is None else vasoactive_signal)
        except Exception:
            vaso = float(demand)

        if not np.isfinite(activity):
            activity = 0.0
        if not np.isfinite(demand):
            demand = 0.0
        if not np.isfinite(vaso):
            vaso = 0.0

        activity = float(np.clip(activity, 0.0, 1.0))
        demand = float(np.clip(demand, 0.0, 1.0))
        vaso = float(np.clip(vaso, 0.0, 1.0))

        # 神经血管耦合：活动/代谢需求 -> 血流提升
        activity_factor = 1.0 + vaso * 0.5 * float(self.vascular_reactivity)
        target_flow = self.baseline_flow * max(activity_factor, 0.0)
        
        # 血流调节（简化的动力学）
        tau_flow = 2000.0  # ms
        self.current_flow += (target_flow - self.current_flow) / tau_flow * dt
        self.current_flow = max(0.0, float(self.current_flow))

        flow_ratio = self.current_flow / max(self.baseline_flow, 1e-9)
        # 归一化供给（用于代谢模型的闭环耦合）
        oxygen_delivery = float(flow_ratio)
        glucose_delivery = float(flow_ratio)
        
        # 氧气和葡萄糖消耗
        oxygen_consumption = oxygen_delivery * (0.5 + 0.5 * demand) * float(self.oxygen_extraction)
        glucose_consumption = float(self.glucose_consumption) * (0.5 + 0.5 * demand)
        
        # BOLD信号计算（简化）
        bold_signal = 0.3 * (flow_ratio - 1.0) * (1.0 - float(self.oxygen_extraction))
        
        return {
            'blood_flow': self.current_flow,
            'flow_ratio': float(flow_ratio),
            'oxygen_delivery': oxygen_delivery,
            'glucose_delivery': glucose_delivery,
            'oxygen_consumption': oxygen_consumption,
            'glucose_consumption': glucose_consumption,
            'bold_signal': bold_signal,
            'vascular_reactivity': self.vascular_reactivity
        }

class MetabolismModel:
    """代谢模型"""
    
    def __init__(self, neuron_count: int):
        self.neuron_count = neuron_count
        
        # 代谢参数
        self.atp_concentration = 2.5  # mM
        self.glucose_concentration = 1.0  # mM
        self.oxygen_concentration = 0.1  # mM
        
        # 消耗速率
        self.basal_atp_consumption = 0.1  # mM/s per neuron
        self.activity_atp_factor = 10.0
        
        # 产生速率
        self.glucose_to_atp_efficiency = 30.0  # ATP per glucose
        self.oxygen_to_atp_efficiency = 6.0   # ATP per oxygen
        
    def update(
        self,
        dt: float,
        neural_activity: float,
        *,
        activity_rate_hz: float | None = None,
        oxygen_delivery: float | None = None,
        glucose_delivery: float | None = None,
    ) -> Dict[str, float]:
        """更新代谢状态（粗粒度能量池 + 血流供给闭环）

        参数:
        - dt: 毫秒
        - neural_activity: 归一化活动水平（0..1）
        - activity_rate_hz: 可选的放电率代理（Hz），将映射为 0..1 的活动驱动
        - oxygen_delivery / glucose_delivery: 可选供给因子（>=0），通常来自 BloodFlowModel.flow_ratio
        """

        try:
            dt_ms = float(dt)
        except Exception:
            dt_ms = 0.0
        dt_s = max(dt_ms, 0.0) / 1000.0

        try:
            activity = float(neural_activity)
        except Exception:
            activity = 0.0

        if activity_rate_hz is not None:
            try:
                rate_hz = float(activity_rate_hz)
            except Exception:
                rate_hz = 0.0
            if np.isfinite(rate_hz):
                reference_rate_hz = 20.0
                activity = 1.0 - float(np.exp(-max(rate_hz, 0.0) / max(reference_rate_hz, 1e-6)))

        if not np.isfinite(activity):
            activity = 0.0
        activity = float(np.clip(activity, 0.0, 1.0))

        try:
            oxy = float(1.0 if oxygen_delivery is None else oxygen_delivery)
        except Exception:
            oxy = 1.0
        try:
            glu = float(1.0 if glucose_delivery is None else glucose_delivery)
        except Exception:
            glu = 1.0
        if not np.isfinite(oxy):
            oxy = 1.0
        if not np.isfinite(glu):
            glu = 1.0
        oxy = max(0.0, oxy)
        glu = max(0.0, glu)
        
        # ATP消耗
        # 注意：region 模型在解剖尺度下会有极大的 neuron_count；这里使用下采样后的
        # 粗粒度能量池模型，避免按真实 neuron_count 线性缩放导致能量瞬时耗尽。
        basal_consumption = float(self.basal_atp_consumption) * dt_s
        activity_consumption = basal_consumption * float(self.activity_atp_factor) * activity
        total_atp_consumption = basal_consumption + activity_consumption
        
        # ATP产生（供给闭环）：供给因子 > 1 表示血流增加带来的恢复
        supply_gain = 0.05  # mM/s 等价增益（粗略）
        supply = supply_gain * ((oxy + glu) / 2.0 - 1.0) * dt_s
        if supply < 0.0:
            supply = 0.0

        # Telemetry-only consumption bookkeeping
        glucose_consumption = total_atp_consumption / max(float(self.glucose_to_atp_efficiency), 1e-6)
        oxygen_consumption = total_atp_consumption / max(float(self.oxygen_to_atp_efficiency), 1e-6)
        
        # 更新浓度
        prev_atp = float(self.atp_concentration)
        self.atp_concentration += float(supply) - float(total_atp_consumption)

        # 物质池：供给提升/活动消耗（粗略）
        self.glucose_concentration += (0.02 * (glu - 1.0) - 0.01 * activity) * dt_s
        self.oxygen_concentration += (0.02 * (oxy - 1.0) - 0.01 * activity) * dt_s
        
        # 限制在生理范围
        self.atp_concentration = max(0.1, self.atp_concentration)
        self.glucose_concentration = max(0.0, self.glucose_concentration)
        self.oxygen_concentration = max(0.0, self.oxygen_concentration)

        energy_deficit = max(0.0, prev_atp - float(self.atp_concentration))
        atp_ratio = float(self.atp_concentration) / max(2.5, 1e-9)
        
        return {
            'atp_concentration': self.atp_concentration,
            'atp_ratio': atp_ratio,
            'glucose_concentration': self.glucose_concentration,
            'oxygen_concentration': self.oxygen_concentration,
            'atp_consumption_rate': total_atp_consumption,
            'glucose_consumption_rate': glucose_consumption,
            'oxygen_consumption_rate': oxygen_consumption,
            'energy_deficit': energy_deficit,
        }
