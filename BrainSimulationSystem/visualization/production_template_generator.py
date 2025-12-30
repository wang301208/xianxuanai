"""
生产级模板生成器

为可视化服务器创建高质量、安全的HTML模板。
包含完整的错误处理、性能优化和安全性考量。
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import time


class ProductionTemplateGenerator:
    """生产级模板生成器"""
    
    def __init__(self, template_dir: Optional[str] = None, brain_system=None):
        """
        初始化模板生成器
        
        Args:
            template_dir: 模板目录路径
            brain_system: 大脑系统实例，用于获取实际数据
        """
        self.logger = logging.getLogger(__name__)
        self.brain_system = brain_system
        
        # 设置模板目录
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent / 'templates'
        
        # 确保模板目录存在
        self._ensure_template_directory()
        
        # 模板缓存
        self._template_cache = {}
        self._cache_timestamps = {}
        
        # 安全配置
        self.security_config = {
            'csp_nonce': self._generate_nonce(),
            'allowed_origins': ['localhost', '127.0.0.1'],
            'max_template_size': 1024 * 1024,  # 1MB
        }
    
    def _ensure_template_directory(self) -> None:
        """确保模板目录存在"""
        try:
            self.template_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"模板目录已准备: {self.template_dir}")
        except Exception as e:
            self.logger.error(f"创建模板目录失败: {e}")
            raise
    
    def _generate_nonce(self) -> str:
        """生成CSP随机数"""
        return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16]
    
    def _get_network_data(self) -> Dict[str, Any]:
        """获取网络数据"""
        try:
            if hasattr(self.brain_system, 'get_network_data'):
                return self.brain_system.get_network_data()
            else:
                # 返回空的数据结构
                return {
                    "neurons": [],
                    "connections": [],
                    "metadata": {
                        "total_neurons": 0,
                        "total_connections": 0,
                        "last_updated": time.time()
                    }
                }
        except Exception as e:
            self.logger.error(f"获取网络数据失败: {e}")
            return {"neurons": [], "connections": [], "metadata": {}}
    
    def _get_cognitive_data(self) -> Dict[str, Any]:
        """获取认知数据"""
        try:
            if hasattr(self.brain_system, 'get_cognitive_state'):
                return self.brain_system.get_cognitive_state()
            else:
                return {
                    "attention": {"focus": [], "intensity": 0.0},
                    "memory": {"working": [], "capacity": 7},
                    "decision": {"state": "idle", "confidence": 0.0}
                }
        except Exception as e:
            self.logger.error(f"获取认知数据失败: {e}")
            return {"attention": {}, "memory": {}, "decision": {}}
    
    def create_all_templates(self) -> bool:
        """创建所有模板"""
        try:
            templates = [
                ('base.html', self._create_base_template),
                ('index.html', self._create_index_template),
                ('visualization.html', self._create_visualization_template),
                ('network.html', self._create_network_template),
                ('cognitive.html', self._create_cognitive_template)
            ]
            
            for template_name, template_func in templates:
                template_path = self.template_dir / template_name
                if not template_path.exists():
                    content = template_func()
                    self._write_template_safely(template_path, content)
                    self.logger.info(f"已创建模板: {template_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"创建模板失败: {e}")
            return False
    
    def _write_template_safely(self, path: Path, content: str) -> None:
        """安全地写入模板文件"""
        try:
            # 检查内容大小
            if len(content.encode('utf-8')) > self.security_config['max_template_size']:
                raise ValueError("模板内容过大")
            
            # 写入文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            self.logger.error(f"写入模板文件失败 {path}: {e}")
            raise
    
    def _create_base_template(self) -> str:
        """创建基础模板"""
        nonce = self.security_config['csp_nonce']
        
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline';">
    <title>{{% block title %}}大脑模拟系统{{% endblock %}}</title>
    <link rel="stylesheet" href="{{{{ url_for('static', filename='css/style.css') }}}}">
    {{% block extra_css %}}{{% endblock %}}
</head>
<body>
    <header>
        <nav class="navbar">
            <div class="navbar-brand">
                <h1>大脑模拟系统</h1>
            </div>
            <ul class="navbar-nav">
                <li><a href="/" class="nav-link">首页</a></li>
                <li><a href="/visualization" class="nav-link">可视化</a></li>
                <li><a href="/network" class="nav-link">神经网络</a></li>
                <li><a href="/cognitive" class="nav-link">认知过程</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="main-content">
        <div class="error-boundary" id="error-boundary">
            {{% block content %}}{{% endblock %}}
        </div>
    </main>
    
    <footer class="footer">
        <div class="footer-content">
            <p>&copy; 2025 大脑模拟系统 - 生产版本</p>
            <div class="footer-links">
                <a href="/api/health">系统状态</a>
                <a href="/api/docs">API文档</a>
            </div>
        </div>
    </footer>
    
    <!-- 全局错误处理 -->
    <script nonce="{nonce}">
        window.addEventListener('error', function(e) {{
            console.error('全局错误:', e.error);
            const errorBoundary = document.getElementById('error-boundary');
            if (errorBoundary && !errorBoundary.classList.contains('has-error')) {{
                errorBoundary.classList.add('has-error');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = '页面发生错误，请刷新重试';
                errorBoundary.appendChild(errorMsg);
            }}
        }});
        
        // 性能监控
        window.addEventListener('load', function() {{
            const perfData = performance.getEntriesByType('navigation')[0];
            if (perfData) {{
                console.log('页面加载时间:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
            }}
        }});
    </script>
    
    <script src="{{{{ url_for('static', filename='js/main.js') }}}}" nonce="{nonce}"></script>
    {{% block extra_js %}}{{% endblock %}}
</body>
</html>"""
    
    def _create_index_template(self) -> str:
        """创建首页模板"""
        return """{{% extends "base.html" %}}

{{% block title %}}大脑模拟系统 - 首页{{% endblock %}}

{{% block content %}}
<div class="container">
    <section class="hero">
        <h1>欢迎使用大脑模拟系统</h1>
        <p class="hero-description">
            高性能神经网络模拟平台，支持实时认知过程分析和可视化
        </p>
    </section>
    
    <section class="features">
        <h2>系统功能</h2>
        <div class="features-grid">
            <div class="feature-card">
                <h3>神经网络模拟</h3>
                <p>高精度神经元网络建模和突触连接仿真</p>
                <ul>
                    <li>多层神经网络架构</li>
                    <li>动态突触可塑性</li>
                    <li>实时活动监控</li>
                </ul>
            </div>
            <div class="feature-card">
                <h3>认知过程分析</h3>
                <p>注意力、记忆和决策过程的深度分析</p>
                <ul>
                    <li>注意力焦点追踪</li>
                    <li>工作记忆容量分析</li>
                    <li>决策路径可视化</li>
                </ul>
            </div>
            <div class="feature-card">
                <h3>神经调质建模</h3>
                <p>多巴胺、血清素等神经调质的影响建模</p>
                <ul>
                    <li>调质浓度监控</li>
                    <li>认知影响分析</li>
                    <li>动态调节机制</li>
                </ul>
            </div>
        </div>
    </section>
    
    <section class="quick-access">
        <h2>快速访问</h2>
        <div class="access-grid">
            <a href="/visualization" class="access-card primary">
                <div class="card-icon">📊</div>
                <h3>实时可视化</h3>
                <p>查看神经活动和认知过程的实时数据</p>
            </a>
            <a href="/network" class="access-card secondary">
                <div class="card-icon">🧠</div>
                <h3>网络结构</h3>
                <p>探索神经元连接和网络拓扑</p>
            </a>
            <a href="/cognitive" class="access-card tertiary">
                <div class="card-icon">💭</div>
                <h3>认知分析</h3>
                <p>深入分析认知过程和决策机制</p>
            </a>
        </div>
    </section>
    
    <section class="system-dashboard">
        <h2>系统仪表板</h2>
        <div class="dashboard-grid">
            <div class="dashboard-card">
                <h3>模拟状态</h3>
                <div class="status-indicator">
                    <span id="simulation-status" class="status-value">检查中...</span>
                    <div class="status-controls">
                        <button id="start-simulation" class="btn btn-primary">启动</button>
                        <button id="stop-simulation" class="btn btn-secondary" disabled>停止</button>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-card">
                <h3>性能指标</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="metric-label">CPU使用率</span>
                        <span id="cpu-usage" class="metric-value">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">内存使用</span>
                        <span id="memory-usage" class="metric-value">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">神经元数量</span>
                        <span id="neuron-count" class="metric-value">-</span>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-card">
                <h3>认知状态</h3>
                <div class="cognitive-overview">
                    <div class="cognitive-item">
                        <span class="label">注意力强度:</span>
                        <div class="progress-bar">
                            <div id="attention-level" class="progress" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="cognitive-item">
                        <span class="label">记忆负载:</span>
                        <div class="progress-bar">
                            <div id="memory-load" class="progress" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
</div>
{{% endblock %}}

{{% block extra_js %}}
<script>
    class SystemDashboard {{
        constructor() {{
            this.updateInterval = null;
            this.init();
        }}
        
        init() {{
            this.bindEvents();
            this.startUpdates();
        }}
        
        bindEvents() {{
            const startBtn = document.getElementById('start-simulation');
            const stopBtn = document.getElementById('stop-simulation');
            
            if (startBtn) {{
                startBtn.addEventListener('click', () => this.startSimulation());
            }}
            
            if (stopBtn) {{
                stopBtn.addEventListener('click', () => this.stopSimulation());
            }}
        }}
        
        async startSimulation() {{
            try {{
                const response = await fetch('/api/simulation/start', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ steps: 1000, interval: 0.1 }})
                }});
                
                const data = await response.json();
                if (data.status === 'started') {{
                    this.updateSimulationStatus(true);
                }}
            }} catch (error) {{
                console.error('启动模拟失败:', error);
                this.showError('启动模拟失败');
            }}
        }}
        
        async stopSimulation() {{
            try {{
                const response = await fetch('/api/simulation/stop', {{
                    method: 'POST'
                }});
                
                const data = await response.json();
                if (data.status === 'stopped') {{
                    this.updateSimulationStatus(false);
                }}
            }} catch (error) {{
                console.error('停止模拟失败:', error);
                this.showError('停止模拟失败');
            }}
        }}
        
        updateSimulationStatus(running) {{
            const statusEl = document.getElementById('simulation-status');
            const startBtn = document.getElementById('start-simulation');
            const stopBtn = document.getElementById('stop-simulation');
            
            if (statusEl) {{
                statusEl.textContent = running ? '运行中' : '已停止';
                statusEl.className = `status-value ${{running ? 'running' : 'stopped'}}`;
            }}
            
            if (startBtn) startBtn.disabled = running;
            if (stopBtn) stopBtn.disabled = !running;
        }}
        
        async updateMetrics() {{
            try {{
                const [statusRes, metricsRes, cognitiveRes] = await Promise.all([
                    fetch('/api/simulation/status'),
                    fetch('/api/system/metrics'),
                    fetch('/api/cognitive/state')
                ]);
                
                const status = await statusRes.json();
                const metrics = await metricsRes.json();
                const cognitive = await cognitiveRes.json();
                
                this.updateSimulationStatus(status.running);
                this.updateSystemMetrics(metrics);
                this.updateCognitiveState(cognitive);
                
            }} catch (error) {{
                console.error('更新指标失败:', error);
            }}
        }}
        
        updateSystemMetrics(metrics) {{
            const cpuEl = document.getElementById('cpu-usage');
            const memoryEl = document.getElementById('memory-usage');
            const neuronEl = document.getElementById('neuron-count');
            
            if (cpuEl && metrics.cpu_usage !== undefined) {{
                cpuEl.textContent = `${{metrics.cpu_usage.toFixed(1)}}%`;
            }}
            
            if (memoryEl && metrics.memory_usage !== undefined) {{
                memoryEl.textContent = `${{(metrics.memory_usage / 1024 / 1024).toFixed(1)}}MB`;
            }}
            
            if (neuronEl && metrics.neuron_count !== undefined) {{
                neuronEl.textContent = metrics.neuron_count.toLocaleString();
            }}
        }}
        
        updateCognitiveState(cognitive) {{
            const attentionEl = document.getElementById('attention-level');
            const memoryEl = document.getElementById('memory-load');
            
            if (attentionEl && cognitive.attention) {{
                const level = (cognitive.attention.intensity || 0) * 100;
                attentionEl.style.width = `${{level}}%`;
            }}
            
            if (memoryEl && cognitive.memory) {{
                const load = ((cognitive.memory.working || []).length / (cognitive.memory.capacity || 7)) * 100;
                memoryEl.style.width = `${{Math.min(load, 100)}}%`;
            }}
        }}
        
        startUpdates() {{
            this.updateMetrics();
            this.updateInterval = setInterval(() => {{
                this.updateMetrics();
            }}, 2000);
        }}
        
        showError(message) {{
            // 简单的错误提示
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-toast';
            errorDiv.textContent = message;
            document.body.appendChild(errorDiv);
            
            setTimeout(() => {{
                errorDiv.remove();
            }}, 3000);
        }}
    }}
    
    // 页面加载完成后初始化
    document.addEventListener('DOMContentLoaded', () => {{
        new SystemDashboard();
    }});
</script>
{{% endblock %}}"""
    
    def _create_visualization_template(self) -> str:
        """创建可视化页面模板"""
        return """{{% extends "base.html" %}}

{{% block title %}}大脑模拟系统 - 实时可视化{{% endblock %}}

{{% block extra_css %}}
<link rel="stylesheet" href="{{{{ url_for('static', filename='css/visualization.css') }}}}">
{{% endblock %}}

{{% block content %}}
<div class="container">
    <header class="page-header">
        <h1>神经活动实时可视化</h1>
        <div class="header-controls">
            <button id="start-visualization" class="btn btn-primary">开始可视化</button>
            <button id="stop-visualization" class="btn btn-secondary" disabled>停止可视化</button>
            <button id="export-data" class="btn btn-outline">导出数据</button>
        </div>
    </header>
    
    <section class="control-panel">
        <div class="control-group">
            <label for="update-frequency">更新频率:</label>
            <select id="update-frequency">
                <option value="100">10 FPS</option>
                <option value="200" selected>5 FPS</option>
                <option value="500">2 FPS</option>
                <option value="1000">1 FPS</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="visualization-mode">显示模式:</label>
            <select id="visualization-mode">
                <option value="activity">神经活动</option>
                <option value="connections">连接强度</option>
                <option value="neuromodulators">神经调质</option>
                <option value="cognitive">认知状态</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="time-window">时间窗口:</label>
            <input type="range" id="time-window" min="10" max="300" value="60" step="10">
            <span id="time-window-value">60s</span>
        </div>
    </section>
    
    <section class="visualization-grid">
        <div class="viz-panel primary">
            <h2>神经元活动热图</h2>
            <div class="chart-container">
                <canvas id="neural-heatmap" width="800" height="400"></canvas>
                <div class="chart-overlay">
                    <div id="heatmap-stats" class="stats-overlay"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-panel secondary">
            <h2>神经调质浓度</h2>
            <div class="chart-container">
                <canvas id="neuromodulator-chart" width="400" height="300"></canvas>
            </div>
        </div>
        
        <div class="viz-panel secondary">
            <h2>网络连接性</h2>
            <div class="chart-container">
                <canvas id="connectivity-chart" width="400" height="300"></canvas>
            </div>
        </div>
        
        <div class="viz-panel tertiary">
            <h2>认知状态时序</h2>
            <div class="chart-container">
                <canvas id="cognitive-timeline" width="600" height="200"></canvas>
            </div>
        </div>
    </section>
    
    <section class="metrics-panel">
        <h2>实时指标</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>网络活动</h3>
                <div class="metric-value" id="network-activity">0.0</div>
                <div class="metric-unit">平均激活率</div>
            </div>
            
            <div class="metric-card">
                <h3>同步性</h3>
                <div class="metric-value" id="synchronization">0.0</div>
                <div class="metric-unit">同步指数</div>
            </div>
            
            <div class="metric-card">
                <h3>信息流</h3>
                <div class="metric-value" id="information-flow">0.0</div>
                <div class="metric-unit">bits/s</div>
            </div>
            
            <div class="metric-card">
                <h3>能耗</h3>
                <div class="metric-value" id="energy-consumption">0.0</div>
                <div class="metric-unit">相对单位</div>
            </div>
        </div>
    </section>
</div>
{{% endblock %}}

{{% block extra_js %}}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    class VisualizationManager {{
        constructor() {{
            this.isRunning = false;
            this.updateInterval = null;
            this.charts = {{}};
            this.dataBuffer = {{
                neural: [],
                neuromodulators: [],
                connectivity: [],
                cognitive: []
            }};
            this.maxBufferSize = 1000;
            
            this.init();
        }}
        
        init() {{
            this.setupCharts();
            this.bindEvents();
            this.loadInitialData();
        }}
        
        setupCharts() {{
            // 神经元活动热图
            const heatmapCanvas = document.getElementById('neural-heatmap');
            if (heatmapCanvas) {{
                this.charts.heatmap = new Chart(heatmapCanvas, {{
                    type: 'scatter',
                    data: {{ datasets: [] }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                callbacks: {{
                                    label: (context) => {{
                                        return `神经元 ${{context.dataIndex}}: 活动度 ${{context.parsed.y.toFixed(3)}}`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ title: {{ display: true, text: '时间 (s)' }} }},
                            y: {{ title: {{ display: true, text: '神经元ID' }} }}
                        }}
                    }}
                }});
            }}
            
            // 神经调质图表
            const neuroCanvas = document.getElementById('neuromodulator-chart');
            if (neuroCanvas) {{
                this.charts.neuromodulators = new Chart(neuroCanvas, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [
                            {{
                                label: '多巴胺',
                                borderColor: 'rgb(255, 99, 132)',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                data: []
                            }},
                            {{
                                label: '血清素',
                                borderColor: 'rgb(54, 162, 235)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                data: []
                            }},
                            {{
                                label: '去甲肾上腺素',
                                borderColor: 'rgb(255, 205, 86)',
                                backgroundColor: 'rgba(255, 205, 86, 0.1)',
                                data: []
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ min: 0, max: 1 }}
                        }}
                    }}
                }});
            }}
        }}
        
        bindEvents() {{
            const startBtn = document.getElementById('start-visualization');
            const stopBtn = document.getElementById('stop-visualization');
            const exportBtn = document.getElementById('export-data');
            const frequencySelect = document.getElementById('update-frequency');
            const modeSelect = document.getElementById('visualization-mode');
            const timeWindow = document.getElementById('time-window');
            
            if (startBtn) {{
                startBtn.addEventListener('click', () => this.startVisualization());
            }}
            
            if (stopBtn) {{
                stopBtn.addEventListener('click', () => this.stopVisualization());
            }}
            
            if (exportBtn) {{
                exportBtn.addEventListener('click', () => this.exportData());
            }}
            
            if (frequencySelect) {{
                frequencySelect.addEventListener('change', (e) => {{
                    if (this.isRunning) {{
                        this.stopVisualization();
                        setTimeout(() => this.startVisualization(), 100);
                    }}
                }});
            }}
            
            if (timeWindow) {{
                timeWindow.addEventListener('input', (e) => {{
                    document.getElementById('time-window-value').textContent = `${{e.target.value}}s`;
                }});
            }}
        }}
        
        async loadInitialData() {{
            try {{
                const response = await fetch('/api/visualization/initial-data');
                const data = await response.json();
                this.updateCharts(data);
            }} catch (error) {{
                console.error('加载初始数据失败:', error);
            }}
        }}
        
        startVisualization() {{
            if (this.isRunning) return;
            
            this.isRunning = true;
            const frequency = parseInt(document.getElementById('update-frequency').value);
            
            this.updateInterval = setInterval(() => {{
                this.fetchAndUpdateData();
            }}, frequency);
            
            document.getElementById('start-visualization').disabled = true;
            document.getElementById('stop-visualization').disabled = false;
        }}
        
        stopVisualization() {{
            if (!this.isRunning) return;
            
            this.isRunning = false;
            if (this.updateInterval) {{
                clearInterval(this.updateInterval);
                this.updateInterval = null;
            }}
            
            document.getElementById('start-visualization').disabled = false;
            document.getElementById('stop-visualization').disabled = true;
        }}
        
        async fetchAndUpdateData() {{
            try {{
                const response = await fetch('/api/visualization/realtime-data');
                const data = await response.json();
                
                this.updateDataBuffer(data);
                this.updateCharts(data);
                this.updateMetrics(data);
                
            }} catch (error) {{
                console.error('获取实时数据失败:', error);
                this.stopVisualization();
            }}
        }}
        
        updateDataBuffer(data) {{
            // 更新数据缓冲区
            Object.keys(this.dataBuffer).forEach(key => {{
                if (data[key]) {{
                    this.dataBuffer[key].push(data[key]);
                    if (this.dataBuffer[key].length > this.maxBufferSize) {{
                        this.dataBuffer[key].shift();
                    }}
                }}
            }});
        }}
        
        updateCharts(data) {{
            // 更新神经调质图表
            if (this.charts.neuromodulators && data.neuromodulators) {{
                const chart = this.charts.neuromodulators;
                const now = new Date().toLocaleTimeString();
                
                chart.data.labels.push(now);
                if (chart.data.labels.length > 50) {{
                    chart.data.labels.shift();
                }}
                
                chart.data.datasets.forEach((dataset, index) => {{
                    const values = ['dopamine', 'serotonin', 'norepinephrine'];
                    dataset.data.push(data.neuromodulators[values[index]] || 0);
                    if (dataset.data.length > 50) {{
                        dataset.data.shift();
                    }}
                }});
                
                chart.update('none');
            }}
        }}
        
        updateMetrics(data) {{
            if (data.metrics) {{
                const metrics = data.metrics;
                
                const networkActivity = document.getElementById('network-activity');
                if (networkActivity && metrics.network_activity !== undefined) {{
                    networkActivity.textContent = metrics.network_activity.toFixed(3);
                }}
                
                const synchronization = document.getElementById('synchronization');
                if (synchronization && metrics.synchronization !== undefined) {{
                    synchronization.textContent = metrics.synchronization.toFixed(3);
                }}
                
                const infoFlow = document.getElementById('information-flow');
                if (infoFlow && metrics.information_flow !== undefined) {{
                    infoFlow.textContent = metrics.information_flow.toFixed(1);
                }}
                
                const energy = document.getElementById('energy-consumption');
                if (energy && metrics.energy_consumption !== undefined) {{
                    energy.textContent = metrics.energy_consumption.toFixed(2);
                }}
            }}
        }}
        
        exportData() {{
            try {{
                const exportData = {{
                    timestamp: new Date().toISOString(),
                    buffer: this.dataBuffer,
                    metadata: {{
                        buffer_size: this.maxBufferSize,
                        export_version: '1.0'
                    }}
                }};
                
                const dataStr = JSON.stringify(exportData, null, 2);
                const dataBlob = new Blob([dataStr], {{ type: 'application/json' }});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `brain_visualization_${{Date.now()}}.json`;
                link.click();
                
                URL.revokeObjectURL(url);
                
            }} catch (error) {{
                console.error('导出数据失败:', error);
            }}
        }}
    }}
    
    // 页面加载完成后初始化
    document.addEventListener('DOMContentLoaded', () => {{
        new VisualizationManager();
    }});
</script>
{{% endblock %}}"""
    
    def _create_network_template(self) -> str:
        """创建网络页面模板"""
        network_data = self._get_network_data()
        
        return f"""{{% extends "base.html" %}}

{{% block title %}}大脑模拟系统 - 神经网络{{% endblock %}}

{{% block extra_css %}}
<link rel="stylesheet" href="{{{{ url_for('static', filename='css/network.css') }}}}">
{{% endblock %}}

{{% block content %}}
<div class="container">
    <header class="page-header">
        <h1>神经网络结构分析</h1>
        <div class="header-controls">
            <button id="refresh-network" class="btn btn-primary">刷新网络</button>
            <button id="export-network" class="btn btn-outline">导出网络</button>
            <button id="analyze-topology" class="btn btn-secondary">拓扑分析</button>
        </div>
    </header>
    
    <section class="network-controls">
        <div class="control-group">
            <label for="layout-algorithm">布局算法:</label>
            <select id="layout-algorithm">
                <option value="force">力导向布局</option>
                <option value="hierarchical">层次布局</option>
                <option value="circular">环形布局</option>
                <option value="grid">网格布局</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="node-size">节点大小:</label>
            <input type="range" id="node-size" min="2" max="20" value="8" step="1">
            <span id="node-size-value">8px</span>
        </div>
        
        <div class="control-group">
            <label for="edge-opacity">连接透明度:</label>
            <input type="range" id="edge-opacity" min="0.1" max="1" value="0.6" step="0.1">
            <span id="edge-opacity-value">0.6</span>
        </div>
        
        <div class="control-group">
            <label for="filter-threshold">连接强度阈值:</label>
            <input type="range" id="filter-threshold" min="0" max="1" value="0.1" step="0.05">
            <span id="filter-threshold-value">0.1</span>
        </div>
    </section>
    
    <section class="network-visualization">
        <div class="viz-container">
            <div id="network-graph" class="graph-container">
                <!-- 网络图将在这里渲染 -->
            </div>
            <div class="graph-overlay">
                <div id="selection-info" class="info-panel">
                    <h3>选择信息</h3>
                    <div id="selection-details">未选择任何节点</div>
                </div>
            </div>
        </div>
    </section>
    
    <section class="network-analysis">
        <div class="analysis-grid">
            <div class="analysis-card">
                <h3>网络统计</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">节点数量:</span>
                        <span id="node-count" class="stat-value">{len(network_data.get('neurons', []))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">边数量:</span>
                        <span id="edge-count" class="stat-value">{len(network_data.get('connections', []))}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">平均度:</span>
                        <span id="avg-degree" class="stat-value">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">聚类系数:</span>
                        <span id="clustering-coeff" class="stat-value">-</span>
                    </div>
                </div>
            </div>
            
            <div class="analysis-card">
                <h3>连接性分析</h3>
                <div class="connectivity-metrics">
                    <div class="metric-row">
                        <span class="metric-label">网络密度:</span>
                        <span id="network-density" class="metric-value">-</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">最短路径长度:</span>
                        <span id="avg-path-length" class="metric-value">-</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">小世界系数:</span>
                        <span id="small-world-coeff" class="metric-value">-</span>
                    </div>
                </div>
            </div>
            
            <div class="analysis-card">
                <h3>中心性指标</h3>
                <div class="centrality-list" id="centrality-ranking">
                    <div class="loading">计算中...</div>
                </div>
            </div>
        </div>
    </section>
</div>
{{% endblock %}}

{{% block extra_js %}}
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
    class NetworkAnalyzer {{
        constructor() {{
            this.networkData = {json.dumps(network_data)};
            this.simulation = null;
            this.svg = null;
            this.selectedNode = null;
            
            this.init();
        }}
        
        init() {{
            this.setupVisualization();
            this.bindEvents();
            this.calculateMetrics();
            this.renderNetwork();
        }}
        
        setupVisualization() {{
            const container = document.getElementById('network-graph');
            const width = container.clientWidth;
            const height = container.clientHeight || 600;
            
            this.svg = d3.select('#network-graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // 添加缩放功能
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {{
                    this.svg.select('.network-group')
                        .attr('transform', event.transform);
                }});
            
            this.svg.call(zoom);
            
            // 创建网络组
            this.networkGroup = this.svg.append('g')
                .attr('class', 'network-group');
        }}
        
        bindEvents() {{
            document.getElementById('refresh-network').addEventListener('click', () => {{
                this.refreshNetwork();
            }});
            
            document.getElementById('export-network').addEventListener('click', () => {{
                this.exportNetwork();
            }});
            
            document.getElementById('analyze-topology').addEventListener('click', () => {{
                this.analyzeTopology();
            }});
            
            document.getElementById('layout-algorithm').addEventListener('change', (e) => {{
                this.changeLayout(e.target.value);
            }});
            
            document.getElementById('node-size').addEventListener('input', (e) => {{
                document.getElementById('node-size-value').textContent = `${{e.target.value}}px`;
                this.updateNodeSize(parseInt(e.target.value));
            }});
            
            document.getElementById('edge-opacity').addEventListener('input', (e) => {{
                document.getElementById('edge-opacity-value').textContent = e.target.value;
                this.updateEdgeOpacity(parseFloat(e.target.value));
            }});
        }}
        
        renderNetwork() {{
            if (!this.networkData.neurons || this.networkData.neurons.length === 0) {{
                this.showEmptyState();
                return;
            }}
            
            // 准备数据
            const nodes = this.networkData.neurons.map(n => ({{
                id: n.id,
                x: n.x || Math.random() * 800,
                y: n.y || Math.random() * 600,
                activity: n.activity || 0
            }}));
            
            const links = this.networkData.connections.map(c => ({{
                source: c.source,
                target: c.target,
                weight: c.weight || 0.5
            }}));
            
            // 创建力模拟
            this.simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(50))
                .force('charge', d3.forceManyBody().strength(-100))
                .force('center', d3.forceCenter(400, 300));
            
            // 渲染连接
            const link = this.networkGroup.selectAll('.link')
                .data(links)
                .enter().append('line')
                .attr('class', 'link')
                .style('stroke', '#999')
                .style('stroke-opacity', 0.6)
                .style('stroke-width', d => Math.sqrt(d.weight) * 2);
            
            // 渲染节点
            const node = this.networkGroup.selectAll('.node')
                .data(nodes)
                .enter().append('circle')
                .attr('class', 'node')
                .attr('r', 8)
                .style('fill', d => this.getNodeColor(d.activity))
                .style('stroke', '#fff')
                .style('stroke-width', 2)
                .on('click', (event, d) => this.selectNode(d))
                .call(d3.drag()
                    .on('start', (event, d) => this.dragStarted(event, d))
                    .on('drag', (event, d) => this.dragged(event, d))
                    .on('end', (event, d) => this.dragEnded(event, d)));
            
            // 添加标签
            const label = this.networkGroup.selectAll('.label')
                .data(nodes)
                .enter().append('text')
                .attr('class', 'label')
                .text(d => d.id)
                .style('font-size', '10px')
                .style('text-anchor', 'middle')
                .style('pointer-events', 'none');
            
            // 更新位置
            this.simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y + 15);
            }});
        }}
        
        getNodeColor(activity) {{
            const intensity = Math.max(0, Math.min(1, activity));
            const hue = (1 - intensity) * 240; // 从蓝色到红色
            return `hsl(${{hue}}, 70%, 50%)`;
        }}
        
        selectNode(node) {{
            this.selectedNode = node;
            this.updateSelectionInfo(node);
            
            // 高亮选中的节点
            this.networkGroup.selectAll('.node')
                .style('stroke', d => d === node ? '#ff0000' : '#fff')
                .style('stroke-width', d => d === node ? 3 : 2);
        }}
        
        updateSelectionInfo(node) {{
            const infoPanel = document.getElementById('selection-details');
            if (node) {{
                infoPanel.innerHTML = `
                    <div class="node-info">
                        <h4>节点: ${{node.id}}</h4>
                        <p>活动度: ${{node.activity.toFixed(3)}}</p>
                        <p>位置: (${{node.x.toFixed(1)}}, ${{node.y.toFixed(1)}})</p>
                    </div>
                `;
            }} else {{
                infoPanel.textContent = '未选择任何节点';
            }}
        }}
        
        calculateMetrics() {{
            if (!this.networkData.neurons || this.networkData.neurons.length === 0) {{
                return;
            }}
            
            const nodeCount = this.networkData.neurons.length;
            const edgeCount = this.networkData.connections.length;
            const avgDegree = edgeCount * 2 / nodeCount;
            
            document.getElementById('node-count').textContent = nodeCount;
            document.getElementById('edge-count').textContent = edgeCount;
            document.getElementById('avg-degree').textContent = avgDegree.toFixed(2);
            
            // 计算网络密度
            const maxEdges = nodeCount * (nodeCount - 1) / 2;
            const density = edgeCount / maxEdges;
            document.getElementById('network-density').textContent = density.toFixed(3);
        }}
        
        async refreshNetwork() {{
            try {{
                const response = await fetch('/api/network/current-state');
                const newData = await response.json();
                
                this.networkData = newData;
                this.clearVisualization();
                this.calculateMetrics();
                this.renderNetwork();
                
            }} catch (error) {{
                console.error('刷新网络失败:', error);
            }}
        }}
        
        clearVisualization() {{
            if (this.simulation) {{
                this.simulation.stop();
            }}
            this.networkGroup.selectAll('*').remove();
        }}
        
        exportNetwork() {{
            try {{
                const exportData = {{
                    timestamp: new Date().toISOString(),
                    network: this.networkData,
                    metadata: {{
                        export_version: '1.0',
                        node_count: this.networkData.neurons.length,
                        edge_count: this.networkData.connections.length
                    }}
                }};
                
                const dataStr = JSON.stringify(exportData, null, 2);
                const dataBlob = new Blob([dataStr], {{ type: 'application/json' }});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `network_structure_${{Date.now()}}.json`;
                link.click();
                
                URL.revokeObjectURL(url);
                
            }} catch (error) {{
                console.error('导出网络失败:', error);
            }}
        }}
        
        showEmptyState() {{
            this.networkGroup.append('text')
                .attr('x', 400)
                .attr('y', 300)
                .attr('text-anchor', 'middle')
                .style('font-size', '18px')
                .style('fill', '#666')
                .text('暂无网络数据');
        }}
        
        // 拖拽事件处理
        dragStarted(event, d) {{
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        dragEnded(event, d) {{
            if (!event.active) this.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    }}
    
    // 页面加载完成后初始化
    document.addEventListener('DOMContentLoaded', () => {{
        new NetworkAnalyzer();
    }});
</script>
{{% endblock %}}"""
    
    def _create_cognitive_template(self) -> str:
        """创建认知页面模板"""
        cognitive_data = self._get_cognitive_data()
        
        return f"""{{% extends "base.html" %}}

{{% block title %}}大脑模拟系统 - 认知过程{{% endblock %}}

{{% block extra_css %}}
<link rel="stylesheet" href="{{{{ url_for('static', filename='css/cognitive.css') }}}}">
{{% endblock %}}

{{% block content %}}
<div class="container">
    <header class="page-header">
        <h1>认知过程深度分析</h1>
        <div class="header-controls">
            <button id="start-cognitive-analysis" class="btn btn-primary">开始分析</button>
            <button id="stop-cognitive-analysis" class="btn btn-secondary" disabled>停止分析</button>
            <button id="export-cognitive-data" class="btn btn-outline">导出数据</button>
        </div>
    </header>
    
    <section class="cognitive-controls">
        <div class="control-group">
            <label for="analysis-mode">分析模式:</label>
            <select id="analysis-mode">
                <option value="realtime">实时分析</option>
                <option value="batch">批量分析</option>
                <option value="comparative">对比分析</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="cognitive-focus">关注焦点:</label>
            <select id="cognitive-focus">
                <option value="attention">注意力机制</option>
                <option value="memory">工作记忆</option>
                <option value="decision">决策过程</option>
                <option value="executive">执行控制</option>
                <option value="integrated">综合分析</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="time-resolution">时间分辨率:</label>
            <select id="time-resolution">
                <option value="100">100ms</option>
                <option value="500" selected>500ms</option>
                <option value="1000">1s</option>
                <option value="5000">5s</option>
            </select>
        </div>
    </section>
    
    <section class="cognitive-dashboard">
        <div class="dashboard-grid">
            <div class="cognitive-panel primary">
                <h2>注意力动态</h2>
                <div class="attention-container">
                    <div class="attention-focus" id="attention-focus">
                        <div class="focus-indicator"></div>
                        <div class="focus-strength" id="focus-strength">
                            强度: {cognitive_data.get('attention', {{}}).get('intensity', 0):.2f}
                        </div>
                    </div>
                    <div class="attention-timeline">
                        <canvas id="attention-timeline-chart" width="400" height="150"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="cognitive-panel secondary">
                <h2>工作记忆状态</h2>
                <div class="memory-container">
                    <div class="memory-slots" id="memory-slots">
                        <!-- 工作记忆槽位将在这里渲染 -->
                    </div>
                    <div class="memory-metrics">
                        <div class="metric">
                            <span class="label">容量利用率:</span>
                            <div class="progress-bar">
                                <div id="memory-utilization" class="progress" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="metric">
                            <span class="label">刷新频率:</span>
                            <span id="memory-refresh-rate" class="value">-</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="cognitive-panel secondary">
                <h2>决策过程</h2>
                <div class="decision-container">
                    <div class="decision-state" id="decision-state">
                        <div class="state-indicator {cognitive_data.get('decision', {{}}).get('state', 'idle')}">
                            {cognitive_data.get('decision', {{}}).get('state', 'idle').title()}
                        </div>
                        <div class="confidence-meter">
                            <span class="label">置信度:</span>
                            <div class="meter">
                                <div id="confidence-level" class="meter-fill" 
                                     style="width: {cognitive_data.get('decision', {{}}).get('confidence', 0) * 100}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="decision-options" id="decision-options">
                        <!-- 决策选项将在这里渲染 -->
                    </div>
                </div>
            </div>
            
            <div class="cognitive-panel tertiary">
                <h2>神经调质影响</h2>
                <div class="neuromodulator-effects">
                    <div class="modulator-item">
                        <span class="modulator-name">多巴胺</span>
                        <div class="effect-bar">
                            <div class="effect-level dopamine" style="width: 60%"></div>
                        </div>
                        <span class="effect-value">0.60</span>
                    </div>
                    <div class="modulator-item">
                        <span class="modulator-name">血清素</span>
                        <div class="effect-bar">
                            <div class="effect-level serotonin" style="width: 45%"></div>
                        </div>
                        <span class="effect-value">0.45</span>
                    </div>
                    <div class="modulator-item">
                        <span class="modulator-name">去甲肾上腺素</span>
                        <div class="effect-bar">
                            <div class="effect-level norepinephrine" style="width: 75%"></div>
                        </div>
                        <span class="effect-value">0.75</span>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <section class="cognitive-analysis">
        <div class="analysis-tabs">
            <button class="tab-button active" data-tab="temporal">时序分析</button>
            <button class="tab-button" data-tab="correlation">相关性分析</button>
            <button class="tab-button" data-tab="patterns">模式识别</button>
            <button class="tab-button" data-tab="predictions">预测分析</button>
        </div>
        
        <div class="tab-content">
            <div id="temporal-tab" class="tab-panel active">
                <h3>认知过程时序分析</h3>
                <div class="temporal-chart-container">
                    <canvas id="temporal-analysis-chart" width="800" height="300"></canvas>
                </div>
            </div>
            
            <div id="correlation-tab" class="tab-panel">
                <h3>认知功能相关性矩阵</h3>
                <div class="correlation-matrix" id="correlation-matrix">
                    <!-- 相关性矩阵将在这里渲染 -->
                </div>
            </div>
            
            <div id="patterns-tab" class="tab-panel">
                <h3>认知模式识别</h3>
                <div class="patterns-container" id="patterns-container">
                    <!-- 模式识别结果将在这里渲染 -->
                </div>
            </div>
            
            <div id="predictions-tab" class="tab-panel">
                <h3>认知状态预测</h3>
                <div class="predictions-container" id="predictions-container">
                    <!-- 预测结果将在这里渲染 -->
                </div>
            </div>
        </div>
    </section>
</div>
{{% endblock %}}

{{% block extra_js %}}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    class CognitiveAnalyzer {{
        constructor() {{
            this.isAnalyzing = false;
            this.analysisInterval = null;
            this.cognitiveData = {json.dumps(cognitive_data)};
            this.charts = {{}};
            this.dataHistory = {{
                attention: [],
                memory: [],
                decision: [],
                neuromodulators: []
            }};
            
            this.init();
        }}
        
        init() {{
            this.setupCharts();
            this.bindEvents();
            this.updateCognitiveDisplay();
            this.setupTabs();
        }}
        
        setupCharts() {{
            // 注意力时序图表
            const attentionCanvas = document.getElementById('attention-timeline-chart');
            if (attentionCanvas) {{
                this.charts.attention = new Chart(attentionCanvas, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [{{
                            label: '注意力强度',
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            data: [],
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ min: 0, max: 1 }}
                        }},
                        plugins: {{
                            legend: {{ display: false }}
                        }}
                    }}
                }});
            }}
            
            // 时序分析图表
            const temporalCanvas = document.getElementById('temporal-analysis-chart');
            if (temporalCanvas) {{
                this.charts.temporal = new Chart(temporalCanvas, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [
                            {{
                                label: '注意力',
                                borderColor: 'rgb(255, 99, 132)',
                                data: []
                            }},
                            {{
                                label: '工作记忆',
                                borderColor: 'rgb(54, 162, 235)',
                                data: []
                            }},
                            {{
                                label: '决策置信度',
                                borderColor: 'rgb(255, 205, 86)',
                                data: []
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ min: 0, max: 1 }}
                        }}
                    }}
                }});
            }}
        }}
        
        bindEvents() {{
            document.getElementById('start-cognitive-analysis').addEventListener('click', () => {{
                this.startAnalysis();
            }});
            
            document.getElementById('stop-cognitive-analysis').addEventListener('click', () => {{
                this.stopAnalysis();
            }});
            
            document.getElementById('export-cognitive-data').addEventListener('click', () => {{
                this.exportData();
            }});
            
            document.getElementById('cognitive-focus').addEventListener('change', (e) => {{
                this.changeFocus(e.target.value);
            }});
        }}
        
        setupTabs() {{
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabPanels = document.querySelectorAll('.tab-panel');
            
            tabButtons.forEach(button => {{
                button.addEventListener('click', () => {{
                    const tabId = button.dataset.tab;
                    
                    // 更新按钮状态
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');
                    
                    // 更新面板显示
                    tabPanels.forEach(panel => {{
                        panel.classList.remove('active');
                        if (panel.id === `${{tabId}}-tab`) {{
                            panel.classList.add('active');
                        }}
                    }});
                }});
            }});
        }}
        
        startAnalysis() {{
            if (this.isAnalyzing) return;
            
            this.isAnalyzing = true;
            const resolution = parseInt(document.getElementById('time-resolution').value);
            
            this.analysisInterval = setInterval(() => {{
                this.updateCognitiveData();
            }}, resolution);
            
            document.getElementById('start-cognitive-analysis').disabled = true;
            document.getElementById('stop-cognitive-analysis').disabled = false;
        }}
        
        stopAnalysis() {{
            if (!this.isAnalyzing) return;
            
            this.isAnalyzing = false;
            if (this.analysisInterval) {{
                clearInterval(this.analysisInterval);
                this.analysisInterval = null;
            }}
            
            document.getElementById('start-cognitive-analysis').disabled = false;
            document.getElementById('stop-cognitive-analysis').disabled = true;
        }}
        
        async updateCognitiveData() {{
            try {{
                const response = await fetch('/api/cognitive/realtime-state');
                const newData = await response.json();
                
                this.cognitiveData = newData;
                this.updateDataHistory(newData);
                this.updateCognitiveDisplay();
                this.updateCharts();
                
            }} catch (error) {{
                console.error('更新认知数据失败:', error);
                this.stopAnalysis();
            }}
        }}
        
        updateDataHistory(data) {{
            const timestamp = new Date().toLocaleTimeString();
            
            // 更新历史数据
            this.dataHistory.attention.push({{
                time: timestamp,
                value: data.attention?.intensity || 0
            }});
            
            this.dataHistory.memory.push({{
                time: timestamp,
                value: (data.memory?.working?.length || 0) / (data.memory?.capacity || 7)
            }});
            
            this.dataHistory.decision.push({{
                time: timestamp,
                value: data.decision?.confidence || 0
            }});
            
            // 限制历史数据长度
            const maxHistory = 100;
            Object.keys(this.dataHistory).forEach(key => {{
                if (this.dataHistory[key].length > maxHistory) {{
                    this.dataHistory[key].shift();
                }}
            }});
        }}
        
        updateCognitiveDisplay() {{
            // 更新注意力显示
            const focusStrength = document.getElementById('focus-strength');
            if (focusStrength && this.cognitiveData.attention) {{
                const intensity = this.cognitiveData.attention.intensity || 0;
                focusStrength.textContent = `强度: ${{intensity.toFixed(2)}}`;
            }}
            
            // 更新工作记忆显示
            this.updateMemorySlots();
            
            // 更新决策状态
            this.updateDecisionState();
            
            // 更新神经调质显示
            this.updateNeuromodulatorEffects();
        }}
        
        updateMemorySlots() {{
            const slotsContainer = document.getElementById('memory-slots');
            if (!slotsContainer || !this.cognitiveData.memory) return;
            
            const memory = this.cognitiveData.memory;
            const capacity = memory.capacity || 7;
            const working = memory.working || [];
            
            slotsContainer.innerHTML = '';
            
            for (let i = 0; i < capacity; i++) {{
                const slot = document.createElement('div');
                slot.className = 'memory-slot';
                
                if (i < working.length) {{
                    slot.classList.add('occupied');
                    slot.textContent = working[i].substring(0, 3) + '...';
                    slot.title = working[i];
                }} else {{
                    slot.classList.add('empty');
                }}
                
                slotsContainer.appendChild(slot);
            }}
            
            // 更新利用率
            const utilization = (working.length / capacity) * 100;
            const utilizationBar = document.getElementById('memory-utilization');
            if (utilizationBar) {{
                utilizationBar.style.width = `${{utilization}}%`;
            }}
        }}
        
        updateDecisionState() {{
            const stateIndicator = document.querySelector('.state-indicator');
            const confidenceLevel = document.getElementById('confidence-level');
            
            if (stateIndicator && this.cognitiveData.decision) {{
                const state = this.cognitiveData.decision.state || 'idle';
                stateIndicator.className = `state-indicator ${{state}}`;
                stateIndicator.textContent = state.charAt(0).toUpperCase() + state.slice(1);
            }}
            
            if (confidenceLevel && this.cognitiveData.decision) {{
                const confidence = (this.cognitiveData.decision.confidence || 0) * 100;
                confidenceLevel.style.width = `${{confidence}}%`;
            }}
        }}
        
        updateNeuromodulatorEffects() {{
            // 这里可以添加神经调质效果的更新逻辑
            // 目前使用静态数据作为示例
        }}
        
        updateCharts() {{
            // 更新注意力时序图表
            if (this.charts.attention && this.dataHistory.attention.length > 0) {{
                const chart = this.charts.attention;
                const recentData = this.dataHistory.attention.slice(-20);
                
                chart.data.labels = recentData.map(d => d.time);
                chart.data.datasets[0].data = recentData.map(d => d.value);
                chart.update('none');
            }}
            
            // 更新时序分析图表
            if (this.charts.temporal) {{
                const chart = this.charts.temporal;
                const recentData = this.dataHistory.attention.slice(-50);
                
                chart.data.labels = recentData.map(d => d.time);
                chart.data.datasets[0].data = this.dataHistory.attention.slice(-50).map(d => d.value);
                chart.data.datasets[1].data = this.dataHistory.memory.slice(-50).map(d => d.value);
                chart.data.datasets[2].data = this.dataHistory.decision.slice(-50).map(d => d.value);
                chart.update('none');
            }}
        }}
        
        exportData() {{
            try {{
                const exportData = {{
                    timestamp: new Date().toISOString(),
                    current_state: this.cognitiveData,
                    history: this.dataHistory,
                    metadata: {{
                        export_version: '1.0',
                        analysis_duration: this.dataHistory.attention.length,
                        sampling_rate: document.getElementById('time-resolution').value
                    }}
                }};
                
                const dataStr = JSON.stringify(exportData, null, 2);
                const dataBlob = new Blob([dataStr], {{ type: 'application/json' }});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `cognitive_analysis_${{Date.now()}}.json`;
                link.click();
                
                URL.revokeObjectURL(url);
                
            }} catch (error) {{
                console.error('导出认知数据失败:', error);
            }}
        }}
    }}
    
    // 页面加载完成后初始化
    document.addEventListener('DOMContentLoaded', () => {{
        new CognitiveAnalyzer();
    }});
</script>
{{% endblock %}}"""

    def get_template_path(self, template_name: str) -> Path:
        """获取模板文件路径"""
        return self.template_dir / template_name

    def template_exists(self, template_name: str) -> bool:
        """检查模板是否存在"""
        return self.get_template_path(template_name).exists()