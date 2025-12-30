# -*- coding: utf-8 -*-
"""
身体模型系统
Body Model System

实现完整的身体模型：
1. 肌肉骨骼系统
2. 运动控制
3. 身体图式
4. 感觉运动整合
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

class JointType(Enum):
    """关节类型"""
    REVOLUTE = "revolute"      # 旋转关节
    PRISMATIC = "prismatic"    # 滑动关节
    SPHERICAL = "spherical"    # 球形关节
    FIXED = "fixed"           # 固定关节

class MuscleType(Enum):
    """肌肉类型"""
    FLEXOR = "flexor"         # 屈肌
    EXTENSOR = "extensor"     # 伸肌
    ABDUCTOR = "abductor"     # 外展肌
    ADDUCTOR = "adductor"     # 内收肌

@dataclass
class Joint:
    """关节定义"""
    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limits: Tuple[float, float] = (-np.pi, np.pi)
    current_angle: float = 0.0
    angular_velocity: float = 0.0
    torque: float = 0.0
    stiffness: float = 1.0
    damping: float = 0.1

@dataclass
class Link:
    """连杆定义"""
    name: str
    mass: float
    length: float
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class Muscle:
    """肌肉定义"""
    name: str
    muscle_type: MuscleType
    origin_link: str
    insertion_link: str
    max_force: float
    optimal_length: float
    current_length: float = 0.0
    activation: float = 0.0
    force: float = 0.0
    fatigue: float = 0.0

class MusculoskeletalModel:
    """肌肉骨骼模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("MusculoskeletalModel")
        
        # 身体结构
        self.links: Dict[str, Link] = {}
        self.joints: Dict[str, Joint] = {}
        self.muscles: Dict[str, Muscle] = {}
        
        # 运动学链
        self.kinematic_chains = {
            'left_arm': ['shoulder_left', 'elbow_left', 'wrist_left'],
            'right_arm': ['shoulder_right', 'elbow_right', 'wrist_right'],
            'left_leg': ['hip_left', 'knee_left', 'ankle_left'],
            'right_leg': ['hip_right', 'knee_right', 'ankle_right'],
            'spine': ['neck', 'thoracic', 'lumbar']
        }
        
        # 物理参数
        self.gravity = np.array([0, 0, -9.81])
        self.dt = 0.001  # 时间步长
        
        # 控制系统
        self.motor_commands = {}
        self.sensory_feedback = {}
        
        self._initialize_body_model()
    
    def _initialize_body_model(self):
        """初始化身体模型"""
        
        # 创建主要连杆
        self._create_links()
        
        # 创建关节
        self._create_joints()
        
        # 创建肌肉
        self._create_muscles()
        
        # 初始化姿态
        self._initialize_posture()
    
    def _create_links(self):
        """创建连杆"""
        
        # 躯干
        self.links['torso'] = Link('torso', mass=40.0, length=0.6)
        self.links['head'] = Link('head', mass=5.0, length=0.25)
        
        # 手臂
        for side in ['left', 'right']:
            self.links[f'upper_arm_{side}'] = Link(f'upper_arm_{side}', mass=3.0, length=0.3)
            self.links[f'forearm_{side}'] = Link(f'forearm_{side}', mass=2.0, length=0.25)
            self.links[f'hand_{side}'] = Link(f'hand_{side}', mass=0.5, length=0.15)
        
        # 腿部
        for side in ['left', 'right']:
            self.links[f'thigh_{side}'] = Link(f'thigh_{side}', mass=8.0, length=0.4)
            self.links[f'shank_{side}'] = Link(f'shank_{side}', mass=4.0, length=0.35)
            self.links[f'foot_{side}'] = Link(f'foot_{side}', mass=1.0, length=0.2)
    
    def _create_joints(self):
        """创建关节"""
        
        # 颈部
        self.joints['neck'] = Joint(
            'neck', JointType.SPHERICAL, 'torso', 'head',
            position=np.array([0, 0, 0.3]), limits=(-np.pi/3, np.pi/3)
        )
        
        # 肩关节
        for side in ['left', 'right']:
            x_pos = 0.2 if side == 'left' else -0.2
            self.joints[f'shoulder_{side}'] = Joint(
                f'shoulder_{side}', JointType.SPHERICAL, 'torso', f'upper_arm_{side}',
                position=np.array([x_pos, 0, 0.25]), limits=(-np.pi, np.pi)
            )
        
        # 肘关节
        for side in ['left', 'right']:
            self.joints[f'elbow_{side}'] = Joint(
                f'elbow_{side}', JointType.REVOLUTE, f'upper_arm_{side}', f'forearm_{side}',
                position=np.array([0, 0, -0.3]), limits=(0, np.pi)
            )
        
        # 腕关节
        for side in ['left', 'right']:
            self.joints[f'wrist_{side}'] = Joint(
                f'wrist_{side}', JointType.SPHERICAL, f'forearm_{side}', f'hand_{side}',
                position=np.array([0, 0, -0.25]), limits=(-np.pi/2, np.pi/2)
            )
        
        # 髋关节
        for side in ['left', 'right']:
            x_pos = 0.1 if side == 'left' else -0.1
            self.joints[f'hip_{side}'] = Joint(
                f'hip_{side}', JointType.SPHERICAL, 'torso', f'thigh_{side}',
                position=np.array([x_pos, 0, -0.3]), limits=(-np.pi/2, np.pi/2)
            )
        
        # 膝关节
        for side in ['left', 'right']:
            self.joints[f'knee_{side}'] = Joint(
                f'knee_{side}', JointType.REVOLUTE, f'thigh_{side}', f'shank_{side}',
                position=np.array([0, 0, -0.4]), limits=(0, np.pi)
            )
        
        # 踝关节
        for side in ['left', 'right']:
            self.joints[f'ankle_{side}'] = Joint(
                f'ankle_{side}', JointType.REVOLUTE, f'shank_{side}', f'foot_{side}',
                position=np.array([0, 0, -0.35]), limits=(-np.pi/3, np.pi/3)
            )
    
    def _create_muscles(self):
        """创建肌肉"""
        
        # 手臂肌肉
        for side in ['left', 'right']:
            # 肱二头肌（屈肌）
            self.muscles[f'biceps_{side}'] = Muscle(
                f'biceps_{side}', MuscleType.FLEXOR,
                f'upper_arm_{side}', f'forearm_{side}',
                max_force=500.0, optimal_length=0.25
            )
            
            # 肱三头肌（伸肌）
            self.muscles[f'triceps_{side}'] = Muscle(
                f'triceps_{side}', MuscleType.EXTENSOR,
                f'upper_arm_{side}', f'forearm_{side}',
                max_force=600.0, optimal_length=0.28
            )
            
            # 三角肌
            self.muscles[f'deltoid_{side}'] = Muscle(
                f'deltoid_{side}', MuscleType.ABDUCTOR,
                'torso', f'upper_arm_{side}',
                max_force=800.0, optimal_length=0.15
            )
        
        # 腿部肌肉
        for side in ['left', 'right']:
            # 股四头肌（伸肌）
            self.muscles[f'quadriceps_{side}'] = Muscle(
                f'quadriceps_{side}', MuscleType.EXTENSOR,
                f'thigh_{side}', f'shank_{side}',
                max_force=2000.0, optimal_length=0.35
            )
            
            # 腘绳肌（屈肌）
            self.muscles[f'hamstring_{side}'] = Muscle(
                f'hamstring_{side}', MuscleType.FLEXOR,
                f'thigh_{side}', f'shank_{side}',
                max_force=1500.0, optimal_length=0.32
            )
            
            # 小腿三头肌
            self.muscles[f'gastrocnemius_{side}'] = Muscle(
                f'gastrocnemius_{side}', MuscleType.FLEXOR,
                f'shank_{side}', f'foot_{side}',
                max_force=1200.0, optimal_length=0.25
            )
    
    def _initialize_posture(self):
        """初始化姿态"""
        
        # 设置初始关节角度（直立姿态）
        for joint_name, joint in self.joints.items():
            if 'shoulder' in joint_name:
                joint.current_angle = 0.0  # 手臂自然下垂
            elif 'elbow' in joint_name:
                joint.current_angle = np.pi/6  # 轻微弯曲
            elif 'hip' in joint_name:
                joint.current_angle = 0.0  # 直立
            elif 'knee' in joint_name:
                joint.current_angle = np.pi/12  # 轻微弯曲
            elif 'ankle' in joint_name:
                joint.current_angle = 0.0  # 水平
            else:
                joint.current_angle = 0.0
        
        # 更新连杆位置
        self._update_forward_kinematics()
    
    def update_dynamics(self, dt: float, motor_commands: Dict[str, float]) -> Dict[str, Any]:
        """更新动力学"""
        
        self.dt = dt
        self.motor_commands = motor_commands
        
        # 计算肌肉激活
        muscle_activations = self._compute_muscle_activations(motor_commands)
        
        # 计算肌肉力
        muscle_forces = self._compute_muscle_forces(muscle_activations)
        
        # 计算关节力矩
        joint_torques = self._compute_joint_torques(muscle_forces)
        
        # 动力学积分
        dynamics_result = self._integrate_dynamics(joint_torques)
        
        # 更新运动学
        self._update_forward_kinematics()
        
        # 计算感觉反馈
        sensory_feedback = self._compute_sensory_feedback()
        
        return {
            'muscle_activations': muscle_activations,
            'muscle_forces': muscle_forces,
            'joint_torques': joint_torques,
            'dynamics_result': dynamics_result,
            'sensory_feedback': sensory_feedback,
            'joint_states': self._get_joint_states(),
            'end_effector_positions': self._get_end_effector_positions()
        }
    
    def _compute_muscle_activations(self, motor_commands: Dict[str, float]) -> Dict[str, float]:
        """计算肌肉激活"""
        
        activations = {}
        
        for muscle_name, muscle in self.muscles.items():
            # 从运动命令获取激活信号
            if muscle_name in motor_commands:
                target_activation = motor_commands[muscle_name]
            else:
                # 默认激活（维持姿态）
                if muscle.muscle_type == MuscleType.EXTENSOR:
                    target_activation = 0.1  # 抗重力肌肉基础激活
                else:
                    target_activation = 0.05
            
            # 激活动力学（一阶系统）
            activation_time_constant = 0.05  # 50ms
            activation_rate = (target_activation - muscle.activation) / activation_time_constant
            
            muscle.activation += activation_rate * self.dt
            muscle.activation = np.clip(muscle.activation, 0.0, 1.0)
            
            # 疲劳效应
            if muscle.activation > 0.8:
                muscle.fatigue += 0.001 * self.dt
            else:
                muscle.fatigue = max(0.0, muscle.fatigue - 0.0005 * self.dt)
            
            # 考虑疲劳的有效激活
            effective_activation = muscle.activation * (1.0 - muscle.fatigue)
            activations[muscle_name] = effective_activation
        
        return activations
    
    def _compute_muscle_forces(self, muscle_activations: Dict[str, float]) -> Dict[str, float]:
        """计算肌肉力"""
        
        forces = {}
        
        for muscle_name, muscle in self.muscles.items():
            activation = muscle_activations[muscle_name]
            
            # 更新肌肉长度
            muscle.current_length = self._compute_muscle_length(muscle)
            
            # 力-长度关系
            length_ratio = muscle.current_length / muscle.optimal_length
            if 0.5 <= length_ratio <= 1.5:
                # 高斯型力-长度关系
                length_factor = np.exp(-2 * (length_ratio - 1.0)**2)
            else:
                length_factor = 0.1  # 超出范围时力显著下降
            
            # 力-速度关系（简化）
            velocity_factor = 1.0  # 暂时简化为常数
            
            # 计算肌肉力
            muscle.force = activation * muscle.max_force * length_factor * velocity_factor
            forces[muscle_name] = muscle.force
        
        return forces
    
    def _compute_muscle_length(self, muscle: Muscle) -> float:
        """计算肌肉长度"""
        
        # 简化的肌肉长度计算
        # 实际应该基于肌肉附着点的3D位置
        
        origin_link = self.links[muscle.origin_link]
        insertion_link = self.links[muscle.insertion_link]
        
        # 计算两个连杆之间的距离
        distance = np.linalg.norm(origin_link.position - insertion_link.position)
        
        return distance
    
    def _compute_joint_torques(self, muscle_forces: Dict[str, float]) -> Dict[str, float]:
        """计算关节力矩"""
        
        torques = {}
        
        for joint_name, joint in self.joints.items():
            total_torque = 0.0
            
            # 查找作用于该关节的肌肉
            for muscle_name, muscle in self.muscles.items():
                if self._muscle_crosses_joint(muscle, joint):
                    # 计算力臂
                    moment_arm = self._compute_moment_arm(muscle, joint)
                    
                    # 计算力矩贡献
                    muscle_force = muscle_forces[muscle_name]
                    
                    # 根据肌肉类型确定力矩方向
                    if muscle.muscle_type == MuscleType.FLEXOR:
                        torque_contribution = muscle_force * moment_arm
                    elif muscle.muscle_type == MuscleType.EXTENSOR:
                        torque_contribution = -muscle_force * moment_arm
                    else:
                        torque_contribution = muscle_force * moment_arm  # 简化
                    
                    total_torque += torque_contribution
            
            # 添加被动力矩（弹性和阻尼）
            passive_torque = self._compute_passive_torque(joint)
            total_torque += passive_torque
            
            joint.torque = total_torque
            torques[joint_name] = total_torque
        
        return torques
    
    def _muscle_crosses_joint(self, muscle: Muscle, joint: Joint) -> bool:
        """判断肌肉是否跨越关节"""
        
        # 简化判断：如果肌肉连接的两个连杆通过该关节相连
        return ((muscle.origin_link == joint.parent_link and muscle.insertion_link == joint.child_link) or
                (muscle.origin_link == joint.child_link and muscle.insertion_link == joint.parent_link))
    
    def _compute_moment_arm(self, muscle: Muscle, joint: Joint) -> float:
        """计算力臂"""
        
        # 简化的力臂计算
        # 实际应该基于肌肉路径和关节轴的几何关系
        
        # 假设力臂与关节角度相关
        angle = joint.current_angle
        
        # 典型的力臂变化模式
        if muscle.muscle_type == MuscleType.FLEXOR:
            moment_arm = 0.05 * (1.0 + 0.5 * np.sin(angle))  # 5cm基础力臂
        elif muscle.muscle_type == MuscleType.EXTENSOR:
            moment_arm = 0.04 * (1.0 + 0.3 * np.cos(angle))  # 4cm基础力臂
        else:
            moment_arm = 0.03  # 默认力臂
        
        return abs(moment_arm)
    
    def _compute_passive_torque(self, joint: Joint) -> float:
        """计算被动力矩"""
        
        # 弹性力矩（关节刚度）
        elastic_torque = -joint.stiffness * joint.current_angle
        
        # 阻尼力矩
        damping_torque = -joint.damping * joint.angular_velocity
        
        # 重力力矩（简化）
        gravity_torque = 0.0
        if 'shoulder' in joint.name or 'hip' in joint.name:
            # 主要关节受重力影响
            child_link = self.links[joint.child_link]
            gravity_torque = child_link.mass * 9.81 * 0.1 * np.sin(joint.current_angle)
        
        return elastic_torque + damping_torque + gravity_torque
    
    def _integrate_dynamics(self, joint_torques: Dict[str, float]) -> Dict[str, Any]:
        """动力学积分"""
        
        dynamics_result = {}
        
        for joint_name, joint in self.joints.items():
            torque = joint_torques[joint_name]
            
            # 简化的动力学方程（忽略惯性耦合）
            # τ = I * α + C * ω + G
            
            # 关节惯量（简化）
            if joint.child_link in self.links:
                child_link = self.links[joint.child_link]
                joint_inertia = child_link.mass * (child_link.length**2) / 12  # 杆的转动惯量
            else:
                joint_inertia = 0.1  # 默认惯量
            
            # 角加速度
            angular_acceleration = torque / joint_inertia
            
            # 积分更新角速度和角度
            joint.angular_velocity += angular_acceleration * self.dt
            joint.current_angle += joint.angular_velocity * self.dt
            
            # 关节限制
            if joint.current_angle < joint.limits[0]:
                joint.current_angle = joint.limits[0]
                joint.angular_velocity = 0.0
            elif joint.current_angle > joint.limits[1]:
                joint.current_angle = joint.limits[1]
                joint.angular_velocity = 0.0
            
            dynamics_result[joint_name] = {
                'angular_acceleration': angular_acceleration,
                'angular_velocity': joint.angular_velocity,
                'angle': joint.current_angle
            }
        
        return dynamics_result
    
    def _update_forward_kinematics(self):
        """更新正向运动学"""
        
        # 从基座开始递归计算每个连杆的位置
        self.links['torso'].position = np.array([0, 0, 1.0])  # 躯干高度1米
        
        # 更新各运动链
        for chain_name, joint_names in self.kinematic_chains.items():
            self._update_kinematic_chain(chain_name, joint_names)
    
    def _update_kinematic_chain(self, chain_name: str, joint_names: List[str]):
        """更新运动学链"""
        
        # 确定起始连杆
        if 'arm' in chain_name:
            base_link_name = 'torso'
        elif 'leg' in chain_name:
            base_link_name = 'torso'
        else:  # spine
            base_link_name = 'torso'
        
        current_position = self.links[base_link_name].position.copy()
        current_orientation = np.array([0, 0, 0])  # 简化为欧拉角
        
        # 沿运动链传播
        for joint_name in joint_names:
            if joint_name in self.joints:
                joint = self.joints[joint_name]
                
                # 更新子连杆位置
                if joint.child_link in self.links:
                    child_link = self.links[joint.child_link]
                    
                    # 简化的变换（仅考虑Z轴旋转）
                    angle = joint.current_angle
                    length = child_link.length
                    
                    # 计算新位置
                    dx = length * np.cos(current_orientation[2] + angle)
                    dy = length * np.sin(current_orientation[2] + angle)
                    dz = 0  # 简化为2D
                    
                    child_link.position = current_position + np.array([dx, dy, dz])
                    
                    # 更新当前位置和方向
                    current_position = child_link.position
                    current_orientation[2] += angle
    
    def _compute_sensory_feedback(self) -> Dict[str, Any]:
        """计算感觉反馈"""
        
        # 本体感觉反馈
        proprioceptive_feedback = {}
        for joint_name, joint in self.joints.items():
            proprioceptive_feedback[joint_name] = {
                'angle': joint.current_angle,
                'angular_velocity': joint.angular_velocity,
                'torque': joint.torque
            }
        
        # 肌肉感觉反馈
        muscle_feedback = {}
        for muscle_name, muscle in self.muscles.items():
            muscle_feedback[muscle_name] = {
                'length': muscle.current_length,
                'force': muscle.force,
                'activation': muscle.activation,
                'fatigue': muscle.fatigue
            }
        
        # 触觉反馈（简化）
        tactile_feedback = self._compute_tactile_feedback()
        
        # 前庭反馈（简化）
        vestibular_feedback = self._compute_vestibular_feedback()
        
        return {
            'proprioceptive': proprioceptive_feedback,
            'muscle': muscle_feedback,
            'tactile': tactile_feedback,
            'vestibular': vestibular_feedback
        }
    
    def _compute_tactile_feedback(self) -> Dict[str, Any]:
        """计算触觉反馈"""
        
        # 简化的触觉反馈
        tactile_feedback = {}
        
        # 检查足部接触
        for side in ['left', 'right']:
            foot_link = self.links[f'foot_{side}']
            
            # 如果足部高度接近地面，认为有接触
            if foot_link.position[2] < 0.1:
                contact_force = max(0, 0.1 - foot_link.position[2]) * 1000  # 简化接触力
                tactile_feedback[f'foot_{side}'] = {
                    'contact': True,
                    'force': contact_force,
                    'pressure': contact_force / 0.02  # 假设接触面积0.02m²
                }
            else:
                tactile_feedback[f'foot_{side}'] = {
                    'contact': False,
                    'force': 0.0,
                    'pressure': 0.0
                }
        
        return tactile_feedback
    
    def _compute_vestibular_feedback(self) -> Dict[str, Any]:
        """计算前庭反馈"""
        
        # 简化的前庭反馈
        head_link = self.links['head']
        
        # 线性加速度
        linear_acceleration = head_link.acceleration + self.gravity
        
        # 角加速度（基于颈部关节）
        if 'neck' in self.joints:
            neck_joint = self.joints['neck']
            angular_acceleration = np.array([0, 0, neck_joint.angular_velocity / self.dt])
        else:
            angular_acceleration = np.zeros(3)
        
        return {
            'linear_acceleration': linear_acceleration,
            'angular_acceleration': angular_acceleration,
            'head_orientation': head_link.position  # 简化
        }
    
    def _get_joint_states(self) -> Dict[str, Dict[str, float]]:
        """获取关节状态"""
        
        joint_states = {}
        for joint_name, joint in self.joints.items():
            joint_states[joint_name] = {
                'angle': joint.current_angle,
                'velocity': joint.angular_velocity,
                'torque': joint.torque
            }
        
        return joint_states
    
    def _get_end_effector_positions(self) -> Dict[str, np.ndarray]:
        """获取末端执行器位置"""
        
        end_effectors = {}
        
        # 手部位置
        for side in ['left', 'right']:
            if f'hand_{side}' in self.links:
                end_effectors[f'hand_{side}'] = self.links[f'hand_{side}'].position
        
        # 足部位置
        for side in ['left', 'right']:
            if f'foot_{side}' in self.links:
                end_effectors[f'foot_{side}'] = self.links[f'foot_{side}'].position
        
        # 头部位置
        if 'head' in self.links:
            end_effectors['head'] = self.links['head'].position
        
        return end_effectors
    
    def compute_inverse_kinematics(self, target_positions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算逆运动学"""
        
        # 简化的逆运动学求解
        target_joint_angles = {}
        
        for end_effector, target_pos in target_positions.items():
            if 'hand' in end_effector:
                # 手臂逆运动学
                side = end_effector.split('_')[1]
                joint_angles = self._solve_arm_ik(target_pos, side)
                
                for joint_name, angle in joint_angles.items():
                    target_joint_angles[joint_name] = angle
            
            elif 'foot' in end_effector:
                # 腿部逆运动学
                side = end_effector.split('_')[1]
                joint_angles = self._solve_leg_ik(target_pos, side)
                
                for joint_name, angle in joint_angles.items():
                    target_joint_angles[joint_name] = angle
        
        return target_joint_angles
    
    def _solve_arm_ik(self, target_pos: np.ndarray, side: str) -> Dict[str, float]:
        """求解手臂逆运动学"""
        
        # 简化的2D逆运动学
        shoulder_pos = self.links['torso'].position + np.array([0.2 if side == 'left' else -0.2, 0, 0.25])
        
        # 计算目标相对于肩部的位置
        relative_pos = target_pos - shoulder_pos
        distance = np.linalg.norm(relative_pos[:2])  # 只考虑x-y平面
        
        # 手臂长度
        upper_arm_length = self.links[f'upper_arm_{side}'].length
        forearm_length = self.links[f'forearm_{side}'].length
        
        # 检查可达性
        total_length = upper_arm_length + forearm_length
        if distance > total_length:
            distance = total_length * 0.95  # 限制在可达范围内
        
        # 余弦定理求解肘关节角度
        cos_elbow = (upper_arm_length**2 + forearm_length**2 - distance**2) / (2 * upper_arm_length * forearm_length)
        cos_elbow = np.clip(cos_elbow, -1, 1)
        elbow_angle = np.pi - np.arccos(cos_elbow)
        
        # 求解肩关节角度
        alpha = np.arctan2(relative_pos[1], relative_pos[0])
        beta = np.arccos((upper_arm_length**2 + distance**2 - forearm_length**2) / (2 * upper_arm_length * distance))
        shoulder_angle = alpha - beta
        
        return {
            f'shoulder_{side}': shoulder_angle,
            f'elbow_{side}': elbow_angle,
            f'wrist_{side}': 0.0  # 简化
        }
    
    def _solve_leg_ik(self, target_pos: np.ndarray, side: str) -> Dict[str, float]:
        """求解腿部逆运动学"""
        
        # 简化的腿部逆运动学
        hip_pos = self.links['torso'].position + np.array([0.1 if side == 'left' else -0.1, 0, -0.3])
        
        # 计算目标相对于髋部的位置
        relative_pos = target_pos - hip_pos
        distance = np.linalg.norm(relative_pos)
        
        # 腿部长度
        thigh_length = self.links[f'thigh_{side}'].length
        shank_length = self.links[f'shank_{side}'].length
        
        # 检查可达性
        total_length = thigh_length + shank_length
        if distance > total_length:
            distance = total_length * 0.95
        
        # 余弦定理求解膝关节角度
        cos_knee = (thigh_length**2 + shank_length**2 - distance**2) / (2 * thigh_length * shank_length)
        cos_knee = np.clip(cos_knee, -1, 1)
        knee_angle = np.pi - np.arccos(cos_knee)
        
        # 求解髋关节角度
        alpha = np.arctan2(-relative_pos[2], np.sqrt(relative_pos[0]**2 + relative_pos[1]**2))
        beta = np.arccos((thigh_length**2 + distance**2 - shank_length**2) / (2 * thigh_length * distance))
        hip_angle = alpha + beta
        
        return {
            f'hip_{side}': hip_angle,
            f'knee_{side}': knee_angle,
            f'ankle_{side}': 0.0  # 简化
        }

# 工厂函数
def create_musculoskeletal_model(config: Optional[Dict[str, Any]] = None) -> MusculoskeletalModel:
    """创建肌肉骨骼模型"""
    if config is None:
        config = {}
    
    return MusculoskeletalModel(config)