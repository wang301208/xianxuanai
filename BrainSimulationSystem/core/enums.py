from enum import Enum

class CellType(Enum):
    """完整的细胞类型枚举"""
    # 皮层锥体细胞
    PYRAMIDAL_L2_3 = "pyramidal_l2_3"
    PYRAMIDAL_L5A = "pyramidal_l5a"
    PYRAMIDAL_L5B = "pyramidal_l5b"
    PYRAMIDAL_L6 = "pyramidal_l6"
    
    # 中间神经元
    PV_INTERNEURON = "pv_interneuron"
    SST_INTERNEURON = "sst_interneuron"
    VIP_INTERNEURON = "vip_interneuron"
    LAMP5_INTERNEURON = "lamp5_interneuron"
    CHANDELIER_CELL = "chandelier_cell"
    MARTINOTTI_CELL = "martinotti_cell"
    BASKET_CELL = "basket_cell"
    
    # 胶质细胞
    PROTOPLASMIC_ASTROCYTE = "protoplasmic_astrocyte"
    FIBROUS_ASTROCYTE = "fibrous_astrocyte"
    BERGMANN_GLIA = "bergmann_glia"
    OLIGODENDROCYTE = "oligodendrocyte"
    OPG_CELL = "opg_cell"
    MICROGLIA_RAMIFIED = "microglia_ramified"
    MICROGLIA_ACTIVATED = "microglia_activated"
    
    # 血管细胞
    ENDOTHELIAL_CELL = "endothelial_cell"
    PERICYTE = "pericyte"
    SMOOTH_MUSCLE_CELL = "smooth_muscle_cell"
    PERIVASCULAR_MACROPHAGE = "perivascular_macrophage"

class BrainRegion(Enum):
    """完整的脑区枚举"""
    # 大脑皮层
    PRIMARY_VISUAL_CORTEX = "v1"
    PRIMARY_AUDITORY_CORTEX = "a1"
    PRIMARY_MOTOR_CORTEX = "m1"
    PRIMARY_SOMATOSENSORY_CORTEX = "s1"
    PREFRONTAL_CORTEX = "pfc"
    ANTERIOR_CINGULATE_CORTEX = "acc"
    POSTERIOR_PARIETAL_CORTEX = "ppc"
    TEMPORAL_CORTEX = "tc"
    OCCIPITAL_CORTEX = "oc"
    FRONTAL_CORTEX = "fc"
    
    # 海马结构
    HIPPOCAMPUS_CA1 = "ca1"
    HIPPOCAMPUS_CA3 = "ca3"
    DENTATE_GYRUS = "dg"
    
    # 丘脑
    THALAMUS_VPL = "vpl"
    THALAMUS_LGN = "lgn"
    THALAMUS_MD = "md"
    THALAMUS_VPM = "vpm"
    
    # 基底神经节
    STRIATUM = "str"
    GLOBUS_PALLIDUS = "gp"
    SUBSTANTIA_NIGRA = "sn"
    SUBTHALAMIC_NUCLEUS = "stn"
    
    # 脑干
    LOCUS_COERULEUS = "lc"
    RAPHE_NUCLEI = "rn"
    VENTRAL_TEGMENTAL_AREA = "vta"
    
    # 小脑
    CEREBELLUM_CORTEX = "cb_ctx"
    DEEP_CEREBELLAR_NUCLEI = "dcn"
    
    # 边缘系统
    AMYGDALA = "amg"
    NUCLEUS_ACCUMBENS = "nac"
    SEPTAL_NUCLEI = "sep"
