"""
动态演化可视化监控

提供个性发展和性别内化的实时可视化
"""

import matplotlib.pyplot as plt

class PersonalityMonitor:
    def __init__(self):
        self.history = {t: [] for t in [
            'openness', 'conscientiousness', 
            'extraversion', 'agreeableness',
            'neuroticism'
        ]}
        self.timesteps = []
    
    def add_record(self, traits: Dict, timestep: int):
        for trait, value in traits.items():
            self.history[trait].append(value)
        self.timesteps.append(timestep)
    
    def update(self):
        result = self.system.step()
        if result:
            for trait, value in result['traits'].items():
                self.history['traits'][trait].append(value)
            
            self.history['gender']['biological'].append(result['gender_state']['biological'])
            self.history['gender']['psychological'].append(result['gender_state']['psychological'])
    
    def plot_trajectories(self):
        """绘制特质变化曲线"""
        plt.figure(figsize=(10, 6))
        for trait, values in self.history.items():
            plt.plot(self.timesteps, values, label=trait)
        plt.title('Personality Trait Trajectories')
        plt.xlabel('Time Steps')
        plt.ylabel('Trait Value')
        plt.legend()
        plt.show()
        
    def plot_radar(self):
        """绘制特质雷达图"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # 准备雷达图数据
        labels = list(self.history.keys())
        values = [np.mean(vals[-10:]) for vals in self.history.values()]
        
        # 闭合曲线
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.show()