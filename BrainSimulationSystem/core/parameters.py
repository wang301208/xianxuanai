"""
神经元模型参数
"""
from dataclasses import dataclass
from .enums import CellType

@dataclass
class NeuronParameters:
    """神经元参数数据类"""
    cell_type: CellType
    
    # 基本电生理参数
    membrane_capacitance: float  # pF
    resting_potential: float     # mV
    threshold: float            # mV
    reset_potential: float      # mV
    refractory_period: float    # ms
    
    # 离子通道参数
    na_channel_density: float   # channels/μm²
    k_channel_density: float    # channels/μm²
    ca_channel_density: float   # channels/μm²
    
    # 形态学参数
    soma_diameter: float        # μm
    dendritic_length: float     # μm
    axonal_length: float        # μm
    spine_density: float        # spines/μm
    
    # 代谢参数
    glucose_consumption: float  # μmol/min/g
    oxygen_consumption: float   # μmol/min/g
    atp_production: float       # μmol/min/g

def get_cell_parameters(cell_type: CellType) -> NeuronParameters:
    """
    根据细胞类型从数据库或预定义映射中获取详细参数。
    
    Args:
        cell_type (CellType): 细胞的类型。

    Returns:
        NeuronParameters: 包含该细胞类型所有参数的数据对象。
    """
    # 这是一个示例数据库，实际应用中可能从文件或数据库加载
    cell_params_db = {
        CellType.PYRAMIDAL_L2_3: NeuronParameters(
            cell_type=cell_type,
            membrane_capacitance=281.0, resting_potential=-70.0, threshold=-50.0,
            reset_potential=-65.0, refractory_period=2.0, na_channel_density=120.0,
            k_channel_density=36.0, ca_channel_density=0.5, soma_diameter=15.0,
            dendritic_length=3000.0, axonal_length=8000.0, spine_density=1.2,
            glucose_consumption=0.8, oxygen_consumption=2.4, atp_production=30.0
        ),
        CellType.PV_INTERNEURON: NeuronParameters(
            cell_type=cell_type,
            membrane_capacitance=80.0, resting_potential=-75.0, threshold=-52.0,
            reset_potential=-70.0, refractory_period=1.0, na_channel_density=100.0,
            k_channel_density=80.0, ca_channel_density=0.2, soma_diameter=12.0,
            dendritic_length=1500.0, axonal_length=3000.0, spine_density=0.3,
            glucose_consumption=1.5, oxygen_consumption=4.5, atp_production=45.0
        ),
        # 可以根据需要添加更多细胞类型的默认参数
    }
    
    # 返回特定类型的参数，如果未找到则返回一个默认值
    return cell_params_db.get(cell_type, cell_params_db[CellType.PYRAMIDAL_L2_3])