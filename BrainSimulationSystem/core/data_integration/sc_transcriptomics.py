"""
单细胞转录组数据整合模块

实现基因表达谱到电生理参数的映射
"""

import scanpy as sc
from .io import load_h5ad

class TranscriptomicsIntegrator:
    def __init__(self, species='human'):
        # 物种特异性参考数据集
        self.reference = {
            'human': 'data/allen_human.h5ad',
            'mouse': 'data/allen_mouse.h5ad'
        }
        self.adata = load_h5ad(self.reference[species])
        
    def map_to_channels(self, cell_type: str):
        """将转录组映射到离子通道参数"""
        # 获取标记基因表达
        markers = {
            'Na': ['SCN1A', 'SCN2A'],
            'K': ['KCNA1', 'KCNB1'],
            'Ca': ['CACNA1A', 'CACNB2']
        }
        
        # 计算通道密度系数
        params = {}
        for ch_type, genes in markers.items():
            expr = self.adata[self.adata.obs['cell_type'] == cell_type, genes].X.mean()
            params[f'{ch_type}_density'] = np.log1p(expr.sum())
            
        return params