"""
可视化JS文件生成器
"""

import os


def create_visualization_js(js_dir):
    """创建可视化JS"""
    visualization_js = os.path.join(js_dir, 'visualization.js')
    if not os.path.exists(visualization_js):
        with open(visualization_js, 'w', encoding='utf-8') as f:
            f.write("""// 可视化JS文件
document.addEventListener('DOMContentLoaded', function() {
    console.log('可视化页面已加载');
    
    // 获取元素
    const startBtn = document.getElementById('start-simulation');
    const stopBtn = document.getElementById('stop-simulation');
    const updateIntervalInput = document.getElementById('update-interval');
    const intervalValue = document.getElementById('interval-value');
    const simulationStatus = document.getElementById('simulation-status');
    const currentStep = document.getElementById('current-step');
    const cognitiveState = document.getElementById('cognitive-state');
    const progress = document.getElementById('progress');
    
    // 更新间隔值显示
    updateIntervalInput.addEventListener('input', function() {
        intervalValue.textContent = this.value;
    });
    
    // 初始化图表
    initCharts();
    
    // 绑定按钮事件
    startBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);
    
    // 初始更新状态
    updateStatus();
    
    // 定期更新状态和数据
    let statusInterval;
    let dataInterval;
    
    function startSimulation() {
        const interval = parseInt(updateIntervalInput.value);
        
        fetch('/api/simulation/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ steps: 1000, interval: 0.1 })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                simulationStatus.textContent = '运行中';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
                // 设置定期更新
                statusInterval = setInterval(updateStatus, 1000);
                dataInterval = setInterval(updateData, interval);
            }
        })
        .catch(error => {
            console.error('启动模拟失败:', error);
            showError('启动模拟失败: ' + error.message);
        });
    }
    
    function stopSimulation() {
        fetch('/api/simulation/stop', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                simulationStatus.textContent = '未运行';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                
                // 清除定期更新
                clearInterval(statusInterval);
                clearInterval(dataInterval);
            }
        })
        .catch(error => {
            console.error('停止模拟失败:', error);
            showError('停止模拟失败: ' + error.message);
        });
    }
    
    function updateStatus() {
        fetch('/api/simulation/status')
            .then(response => response.json())
            .then(data => {
                simulationStatus.textContent = data.running ? '运行中' : '未运行';
                startBtn.disabled = data.running;
                stopBtn.disabled = !data.running;
                currentStep.textContent = data.current_step;
                
                // 更新进度条
                const progressPercent = (data.current_step / data.total_steps) * 100;
                progress.style.width = progressPercent + '%';
            })
            .catch(error => {
                console.error('获取模拟状态失败:', error);
            });
            
        fetch('/api/cognitive/state')
            .then(response => response.json())
            .then(data => {
                cognitiveState.textContent = data.cognitive_state || '-';
            })
            .catch(error => {
                console.error('获取认知状态失败:', error);
            });
    }
    
    function updateData() {
        updateNeuralActivityChart();
        updateNeuromodulatorsChart();
        updateAttentionChart();
        updateMemoryContent();
    }
});

// 图表对象
let neuralActivityChart;
let neuromodulatorsChart;
let attentionChart;

// 初始化图表
function initCharts() {
    // 神经元活动图表
    const neuralActivityCtx = document.getElementById('neural-activity-chart').getContext('2d');
    neuralActivityChart = new Chart(neuralActivityCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [{
                label: '兴奋性神经元',
                data: Array(50).fill(0),
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: '抑制性神经元',
                data: Array(50).fill(0),
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            animation: {
                duration: 0
            }
        }
    });
    
    // 神经调质水平图表
    const neuromodulatorsCtx = document.getElementById('neuromodulators-chart').getContext('2d');
    neuromodulatorsChart = new Chart(neuromodulatorsCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [{
                label: '多巴胺',
                data: Array(50).fill(0),
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: '血清素',
                data: Array(50).fill(0),
                borderColor: '#9b59b6',
                backgroundColor: 'rgba(155, 89, 182, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: '乙酰胆碱',
                data: Array(50).fill(0),
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            animation: {
                duration: 0
            }
        }
    });
    
    // 注意力焦点图表
    const attentionCtx = document.getElementById('attention-chart').getContext('2d');
    attentionChart = new Chart(attentionCtx, {
        type: 'radar',
        data: {
            labels: ['视觉', '听觉', '触觉', '内部状态', '记忆检索', '决策'],
            datasets: [{
                label: '注意力分配',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: '#3498db',
                pointBackgroundColor: '#3498db',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#3498db'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}

// 更新神经元活动图表
function updateNeuralActivityChart() {
    fetch('/api/visualization/neural_activity')
        .then(response => response.json())
        .then(data => {
            // 移除第一个数据点并添加新数据点
            neuralActivityChart.data.datasets[0].data.shift();
            neuralActivityChart.data.datasets[0].data.push(data.excitatory);
            
            neuralActivityChart.data.datasets[1].data.shift();
            neuralActivityChart.data.datasets[1].data.push(data.inhibitory);
            
            // 更新图表
            neuralActivityChart.update();
        })
        .catch(error => {
            console.error('获取神经元活动数据失败:', error);
        });
}

// 更新神经调质水平图表
function updateNeuromodulatorsChart() {
    fetch('/api/visualization/neuromodulators')
        .then(response => response.json())
        .then(data => {
            // 移除第一个数据点并添加新数据点
            neuromodulatorsChart.data.datasets[0].data.shift();
            neuromodulatorsChart.data.datasets[0].data.push(data.dopamine);
            
            neuromodulatorsChart.data.datasets[1].data.shift();
            neuromodulatorsChart.data.datasets[1].data.push(data.serotonin);
            
            neuromodulatorsChart.data.datasets[2].data.shift();
            neuromodulatorsChart.data.datasets[2].data.push(data.acetylcholine);
            
            // 更新图表
            neuromodulatorsChart.update();
        })
        .catch(error => {
            console.error('获取神经调质数据失败:', error);
        });
}

// 更新注意力焦点图表
function updateAttentionChart() {
    fetch('/api/visualization/attention')
        .then(response => response.json())
        .then(data => {
            // 更新数据
            attentionChart.data.datasets[0].data = [
                data.visual,
                data.auditory,
                data.tactile,
                data.internal,
                data.memory,
                data.decision
            ];
            
            // 更新图表
            attentionChart.update();
        })
        .catch(error => {
            console.error('获取注意力数据失败:', error);
        });
}

// 更新记忆内容
function updateMemoryContent() {
    const memoryContainer = document.getElementById('memory-content');
    
    fetch('/api/visualization/memory')
        .then(response => response.json())
        .then(data => {
            // 清空容器
            memoryContainer.innerHTML = '';
            
            if (data.items.length === 0) {
                const placeholder = document.createElement('div');
                placeholder.className = 'memory-placeholder';
                placeholder.textContent = '无记忆内容';
                memoryContainer.appendChild(placeholder);
                return;
            }
            
            // 添加记忆项
            data.items.forEach(item => {
                const memoryItem = document.createElement('div');
                memoryItem.className = 'memory-item';
                if (item.active) {
                    memoryItem.classList.add('active');
                }
                
                const content = document.createElement('div');
                content.textContent = item.content;
                
                const strength = document.createElement('span');
                strength.className = 'strength';
                strength.textContent = (item.strength * 100).toFixed(0) + '%';
                
                memoryItem.appendChild(content);
                memoryItem.appendChild(strength);
                memoryContainer.appendChild(memoryItem);
            });
        })
        .catch(error => {
            console.error('获取记忆数据失败:', error);
            
            // 显示错误占位符
            memoryContainer.innerHTML = '';
            const placeholder = document.createElement('div');
            placeholder.className = 'memory-placeholder';
            placeholder.textContent = '无法加载记忆数据';
            memoryContainer.appendChild(placeholder);
        });
}""")