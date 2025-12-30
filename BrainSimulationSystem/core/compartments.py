"""
神经元多室结构创建模块
"""
from typing import Dict, Any, List
from .parameters import NeuronParameters

def create_soma(params: NeuronParameters) -> Dict[str, Any]:
    """
    根据参数创建胞体（soma）室。

    Args:
        params (NeuronParameters): 神经元参数。

    Returns:
        Dict[str, Any]: 描述胞体室属性的字典。
    """
    return {
        "name": "soma",
        "diameter": params.soma_diameter,
        "length": params.soma_diameter,  # 球形近似
        "membrane_potential": params.resting_potential,
        "ion_concentrations": {"Na+": 10, "K+": 140, "Ca2+": 0.0001},
    }

def create_basal_dendrites(params: NeuronParameters, num_dendrites: int = 5) -> List[Dict[str, Any]]:
    """
    创建基底树突（basal dendrites）室。

    Args:
        params (NeuronParameters): 神经元参数。
        num_dendrites (int): 创建的基底树突数量。

    Returns:
        List[Dict[str, Any]]: 描述每个基底树突室的字典列表。
    """
    dendrites = []
    for i in range(num_dendrites):
        dendrites.append({
            "name": f"basal_dendrite_{i}",
            "diameter": params.soma_diameter / 4,
            "length": params.dendritic_length / num_dendrites,
            "membrane_potential": params.resting_potential,
            "ion_concentrations": {"Na+": 10, "K+": 140, "Ca2+": 0.0001},
        })
    return dendrites

def create_apical_dendrite(params: NeuronParameters) -> Dict[str, Any]:
    """
    创建顶端树突（apical dendrite）室。

    Args:
        params (NeuronParameters): 神经元参数。

    Returns:
        Dict[str, Any]: 描述顶端树突室属性的字典。
    """
    return {
        "name": "apical_dendrite",
        "diameter": params.soma_diameter / 2,
        "length": params.dendritic_length * 0.6, # 假设顶突占总长度的60%
        "membrane_potential": params.resting_potential,
        "ion_concentrations": {"Na+": 10, "K+": 140, "Ca2+": 0.0001},
    }

def create_axon(params: NeuronParameters) -> Dict[str, Any]:
    """
    创建轴突（axon）室。

    Args:
        params (NeuronParameters): 神经元参数。

    Returns:
        Dict[str, Any]: 描述轴突室属性的字典。
    """
    return {
        "name": "axon",
        "diameter": params.soma_diameter / 10,
        "length": params.axonal_length,
        "membrane_potential": params.resting_potential,
        "ion_concentrations": {"Na+": 10, "K+": 140, "Ca2+": 0.0001},
    }