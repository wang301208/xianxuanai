"""
多组学参数优化框架

整合转录组/蛋白质组/电生理数据
"""

from typing import Dict
from dataclasses import dataclass

@dataclass
class OmicsProfile:
    transcriptomics: Dict[str, float]
    proteomics: Dict[str, float]
    ephys: Dict[str, float]

class MultiOmicsIntegrator:
    def __init__(self):
        self.weighting = {
            'transcriptome': 0.5,
            'proteome': 0.3,
            'ephys': 0.2
        }
        
    def integrate(self, cell_type: str) -> Dict:
        """健康状态多组学数据整合"""
        # 仅加载健康参考数据
        tx_data = self._get_healthy_transcriptome(cell_type)
        pt_data = self._get_healthy_proteome(cell_type)
        ephys_data = self._get_healthy_ephys(cell_type)
        
        # 标准化健康参数范围
        params = {}
        for k in tx_data.keys():
            params[k] = self._normalize_healthy_range(
                tx_data[k]*self.weighting['transcriptome'] + 
                pt_data[k]*self.weighting['proteome'] + 
                ephys_data[k]*self.weighting['ephys']
            )
        return params
        
    def _normalize_healthy_range(self, value):
        """确保参数在健康范围内"""
        return min(max(value, 0.8), 1.2)  # 保持±20%正常波动