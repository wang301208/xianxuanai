"""
睡眠纺锤波记忆选择器

实现纺锤波-慢波耦合依赖的记忆重放选择
"""

from ..memory_system import HippocampalFormation

class SpindleMemorySelector:
    def __init__(self):
        # 耦合分析参数
        self.coupling_params = {
            'n_cycles': 5,          # 分析周期数
            'bandwidth': (11, 16), # 纺锤波频率带(Hz)
            'phase_bins': 18       # 相位分箱数
        }
        # 相位锁定值缓存
        self.plv_cache = {}
        
        # 注册神经生理组件
        self.hippocampus = HippocampalFormation()
        self.phase_detector = PhaseAnalyzer()
        
        # 动态阈值参数
        self.base_threshold = 0.5
        self.plv_sensitivity = 0.3
        
    def select_for_replay(self, spindle_data):
        """基于纺锤波特性选择记忆"""
        if spindle_data['power'] < self.spindle_power_threshold:
            return []
            
        # 获取候选记忆
        candidates = [
            t for t in self.hippocampus.memory_traces
            if t['strength'] > 0.5
        ]
        
        # 计算相位耦合得分
        scored = []
        for mem in candidates:
            score = self._calculate_coupling_score(
                mem['theta_power'],
                spindle_data['phase']
            )
            scored.append((mem, score))
            
        # 选择Top 20%
        scored.sort(key=lambda x: -x[1])
        return [m for m,_ in scored[:int(0.2*len(scored))]]
        
    def _calculate_phase_coupling(self, slow_wave, spindle):
        """基于希尔伯特变换的相位耦合分析"""
        # 提取相位信息
        slow_phase = np.angle(hilbert(slow_wave))
        spindle_phase = np.angle(hilbert(spindle))
        
        # 计算相位差分布
        phase_diff = slow_phase - spindle_phase
        hist, _ = np.histogram(
            phase_diff, 
            bins=self.coupling_params['phase_bins'],
            range=(-np.pi, np.pi)
        )
        
        # 计算相位锁定值(PLV)
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        return plv, hist
        
    def get_phase_lock_value(self, mem_id):
        """带缓存的相位锁定值查询"""
        if mem_id not in self.plv_cache:
            trace = self.hippocampus.get_trace(mem_id)
            self.plv_cache[mem_id] = self._calculate_plv(trace)
        return self.plv_cache[mem_id]