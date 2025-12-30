"""
完整详细的突触模型
Complete and Detailed Synapse Model
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from .synapse_types import NeurotransmitterType, VesiclePool, ReceptorType, ReceptorKinetics
from .receptor import DetailedReceptor
from .neurotransmitter import NeurotransmitterDynamics
from .plasticity import ShortTermPlasticity, LongTermPlasticity

class CompleteSynapse:
    """
    完整而详细的突触模型，整合了神经递质、囊泡、受体和可塑性动力学。
    A complete and detailed synapse model, integrating neurotransmitter, vesicle, receptor, and plasticity dynamics.
    """
    
    def __init__(self, pre_neuron_id: int, post_neuron_id: int, 
                 synapse_config: Dict[str, Any]):
        """
        初始化一个完整的突触。

        Args:
            pre_neuron_id (int): 突触前神经元的ID。
            post_neuron_id (int): 突触后神经元的ID。
            synapse_config (Dict[str, Any]): 包含突触所有配置的字典。
        """
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.config = synapse_config
        self.logger = logging.getLogger(f"Synapse_{pre_neuron_id}_{post_neuron_id}")
        
        # 基本参数
        self.initial_weight = synapse_config.get('weight', 1.0)
        self.current_weight = self.initial_weight
        self.delay = synapse_config.get('delay', 1.0)
        
        # 初始化各个子模块
        self.nt_type = NeurotransmitterType(synapse_config.get('neurotransmitter', 'glutamate'))
        self.nt_dynamics = NeurotransmitterDynamics(self.nt_type)
        self.vesicle_pool = self._initialize_vesicle_pool(synapse_config)
        self.receptors = self._initialize_receptors(synapse_config.get('receptors', {}))
        self.stp = self._initialize_stp(synapse_config)
        self.ltp = self._initialize_ltp(synapse_config)
        
        # 延迟队列，用于处理突触延迟
        self.spike_queue: List[float] = []
        
        # 状态变量
        self.total_spikes_processed = 0
        self.last_spike_time = -np.inf
        # Latest neuromodulator snapshot influencing STP/LTP dynamics.
        self.neuromodulators: Dict[str, float] = {}

        self.current_psc = 0.0  # 突触后电流 (Postsynaptic Current)
        
    def _initialize_vesicle_pool(self, config: Dict[str, Any]) -> VesiclePool:
        return VesiclePool(
            readily_releasable=config.get('rr_pool_size', 10),
            recycling=config.get('recycling_pool_size', 100),
            reserve=config.get('reserve_pool_size', 1000)
        )

    def _initialize_receptors(self, receptor_config: Dict[str, Any]) -> Dict[ReceptorType, DetailedReceptor]:
        receptors = {}
        # 如果用户没有指定受体，则使用基于神经递质的默认配置
        if not receptor_config:
            receptor_config = self._get_default_receptors()

        for receptor_name, density in receptor_config.items():
            try:
                receptor_type = ReceptorType(receptor_name.lower())
                kinetics = self._get_receptor_kinetics(receptor_type)
                receptors[receptor_type] = DetailedReceptor(receptor_type, density, kinetics)
            except ValueError:
                self.logger.warning(f"未知的受体类型 '{receptor_name}'，已跳过。")
        return receptors

    def _initialize_stp(self, config: Dict[str, Any]) -> ShortTermPlasticity | None:
        if not config.get('stp_enabled', True):
            return None
        return ShortTermPlasticity(
            tau_rec=config.get('tau_rec', 800.0),
            tau_fac=config.get('tau_fac', 50.0),
            U=config.get('U', 0.5)
        )

    def _initialize_ltp(self, config: Dict[str, Any]) -> LongTermPlasticity | None:
        if not config.get('ltp_enabled', True):
            return None
        return LongTermPlasticity(
            learning_rate=config.get('learning_rate', 0.01),
            metaplasticity=config.get('metaplasticity', True)
        )

    def _get_default_receptors(self) -> Dict[str, float]:
        if self.nt_type == NeurotransmitterType.GLUTAMATE:
            return {'ampa': 100.0, 'nmda': 20.0}
        elif self.nt_type == NeurotransmitterType.GABA:
            return {'gaba_a': 200.0}
        return {'ampa': 50.0}

    def _get_receptor_kinetics(self, receptor_type: ReceptorType) -> ReceptorKinetics:
        # 这是一个简化的动力学参数数据库
        kinetics_db = {
            ReceptorType.AMPA: ReceptorKinetics(receptor_type, 10.0, 0.5, 2.0, 0.5, 0.1, 0.01, 20.0, 0.0),
            ReceptorType.NMDA: ReceptorKinetics(receptor_type, 5.0, 0.1, 0.5, 0.1, 0.01, 0.001, 50.0, 0.0, mg_block=True, voltage_dependence=True),
            ReceptorType.GABA_A: ReceptorKinetics(receptor_type, 20.0, 1.0, 5.0, 1.0, 0.05, 0.005, 30.0, -70.0),
            ReceptorType.GABA_B: ReceptorKinetics(receptor_type, 1.0, 0.1, 0.1, 0.05, 0.001, 0.0001, 10.0, -90.0),
        }
        return kinetics_db.get(receptor_type, kinetics_db[ReceptorType.AMPA])

    def process_presynaptic_spike(self, spike_time: float):
        """处理突触前脉冲，将其放入延迟队列。"""
        self.spike_queue.append(spike_time + self.delay)
        self.total_spikes_processed += 1

    def process_postsynaptic_spike(self, spike_time: float):
        """处理突触后脉冲，用于LTP计算。"""
        if self.ltp:
            try:
                self.ltp.apply_neuromodulation(self.neuromodulators)
            except Exception:
                pass
            weight_change = self.ltp.process_post_spike(self.current_weight)
            self.current_weight += weight_change
            self.current_weight = np.clip(self.current_weight, 0.0, 10.0) # 权重限制

    def update(
        self,
        dt: float,
        current_time: float,
        post_membrane_voltage: float,
        astrocyte_activity: float = 0.0,
        neuromodulators: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        在每个时间步更新整个突触的状态。

        Args:
            dt (float): 时间步长 (ms)。
            current_time (float): 当前模拟时间 (ms)。
            post_membrane_voltage (float): 突触后膜电位 (mV)。
            astrocyte_activity (float): 周围星形胶质细胞的活动水平 (归一化值)。

        Returns:
            float: 计算出的总突触后电流 (pA)。
        """
        if neuromodulators is not None and isinstance(neuromodulators, dict):
            self.apply_neuromodulators(neuromodulators)

        # 1. 处理延迟队列中的脉冲
        self._process_spike_queue(current_time)
        
        # 2. 更新神经递质动力学
        astrocyte_uptake_rate = astrocyte_activity * 0.1 # 假设的调节作用
        nt_conc = self.nt_dynamics.update(dt, astrocyte_uptake_rate)
        
        # 3. 更新所有受体并计算总电流
        total_current = 0.0
        for receptor in self.receptors.values():
            total_current += receptor.update(dt, nt_conc, post_membrane_voltage)
            
        # 4. 应用突触权重
        self.current_psc = total_current * self.current_weight
        
        # 5. 更新可塑性状态
        if self.stp:
            try:
                self.stp.apply_neuromodulation(self.neuromodulators)
            except Exception:
                pass
            self.stp.update(dt)
        if self.ltp:
            try:
                self.ltp.apply_neuromodulation(self.neuromodulators)
            except Exception:
                pass
            self.ltp.update(dt)
            
        # 6. 更新囊泡池
        self._update_vesicle_pool(dt)
        
        return self.current_psc

    def apply_neuromodulation(self, modulator_type: str, concentration: float) -> None:
        """Set a single neuromodulator level for this synapse."""
        try:
            key = str(modulator_type)
            value = max(0.0, float(concentration))
        except Exception:
            return
        self.neuromodulators[key] = value

    def apply_neuromodulators(self, neuromodulators: Dict[str, float]) -> None:
        """Replace neuromodulator snapshot used by plasticity dynamics."""
        cleaned: Dict[str, float] = {}
        for key, value in neuromodulators.items():
            try:
                cleaned[str(key)] = max(0.0, float(value))
            except Exception:
                continue
        self.neuromodulators = cleaned

    def _process_spike_queue(self, current_time: float):
        """处理到期的突触前脉冲。"""
        
        # 从队列头部开始处理，直到找到未到期的脉冲
        while self.spike_queue and current_time >= self.spike_queue[0]:
            spike_time = self.spike_queue.pop(0)
            self.last_spike_time = spike_time
            
            # a. 计算释放概率 (受STP影响)
            release_prob = self.nt_dynamics.release_probability
            if self.stp:
                try:
                    self.stp.apply_neuromodulation(self.neuromodulators)
                except Exception:
                    pass
                release_prob = self.stp.process_spike()

            # b. 计算释放的囊泡数
            available_vesicles = self.vesicle_pool.current_rr
            num_released = np.random.binomial(available_vesicles, release_prob)
            
            # c. 更新囊泡池
            self.vesicle_pool.current_rr -= num_released
            
            # d. 触发LTD计算
            if self.ltp:
                try:
                    self.ltp.apply_neuromodulation(self.neuromodulators)
                except Exception:
                    pass
                weight_change = self.ltp.process_pre_spike(self.current_weight)
                self.current_weight += weight_change
                self.current_weight = np.clip(self.current_weight, 0.0, 10.0)

            if num_released > 0:
                self.nt_dynamics.release_vesicles(num_released)

    def _update_vesicle_pool(self, dt: float):
        """更新囊泡池的恢复过程。"""
        pool = self.vesicle_pool
        
        # 从循环池补充即刻可释放池
        refill_amount = (pool.readily_releasable - pool.current_rr) * pool.refill_rate * dt
        refill_amount = min(refill_amount, pool.current_recycling)
        pool.current_rr += refill_amount
        pool.current_recycling -= refill_amount
        
        # 从储备池动员到循环池
        mobilization_amount = (pool.recycling - pool.current_recycling) * pool.mobilization_rate * dt
        mobilization_amount = min(mobilization_amount, pool.current_reserve)
        pool.current_recycling += mobilization_amount
        pool.current_reserve -= mobilization_amount

__all__ = ["CompleteSynapse"]
