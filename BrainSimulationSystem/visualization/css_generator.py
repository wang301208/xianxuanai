"""
CSS生成器

为可视化服务器创建默认CSS文件。
"""

import os


def create_css_files(css_dir):
    """创建所有CSS文件"""
    # 创建基础CSS
    create_base_css(css_dir)
    
    # 创建可视化CSS
    create_visualization_css(css_dir)
    
    # 创建网络CSS
    create_network_css(css_dir)
    
    # 创建认知CSS
    create_cognitive_css(css_dir)


def create_base_css(css_dir):
    """创建基础CSS"""
    base_css = os.path.join(css_dir, 'style.css')
    if not os.path.exists(base_css):
        with open(base_css, 'w', encoding='utf-8') as f:
            f.write("""/* 基础样式 */
:root {
    --primary-color: #3498db;
    --secondary-color: #2c3e50;
    --accent-color: #e74c3c;
    --background-color: #f5f5f5;
    --text-color: #333;
    --border-color: #ddd;
    --success-color: #2ecc71;
    --warning-color: #f39c12;
    --error-color: #e74c3c;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* 头部样式 */
header {
    background-color: var(--secondary-color);
    color: white;
    padding: 10px 0;
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

nav ul {
    display: flex;
    list-style: none;
}

nav ul li {
    margin-left: 20px;
}

nav ul li a {
    color: white;
    text-decoration: none;
    transition: color 0.3s;
}

nav ul li a:hover {
    color: var(--primary-color);
}

/* 主要内容样式 */
main {
    min-height: calc(100vh - 120px);
    padding: 20px 0;
}

h1, h2, h3 {
    margin-bottom: 15px;
    color: var(--secondary-color);
}

h1 {
    font-size: 2rem;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 10px;
    margin-bottom: 20px;
}

h2 {
    font-size: 1.5rem;
}

h3 {
    font-size: 1.2rem;
}

p {
    margin-bottom: 15px;
}

/* 按钮样式 */
.btn {
    display: inline-block;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.3s, transform 0.1s;
}

.btn:hover {
    transform: translateY(-2px);
}

.btn:active {
    transform: translateY(0);
}

.btn.primary {
    background-color: var(--primary-color);
    color: white;
}

.btn.primary:hover {
    background-color: #2980b9;
}

.btn.secondary {
    background-color: var(--secondary-color);
    color: white;
}

.btn.secondary:hover {
    background-color: #1a252f;
}

.btn:disabled {
    background-color: var(--border-color);
    cursor: not-allowed;
    transform: none;
}

/* 表单元素样式 */
input, select, textarea {
    padding: 8px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 1rem;
}

input[type="range"] {
    width: 100%;
}

/* 卡片样式 */
.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}

/* 状态面板样式 */
.status-panel {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-top: 30px;
}

.status-item {
    display: flex;
    margin-bottom: 10px;
    align-items: center;
}

.status-item .label {
    font-weight: bold;
    width: 120px;
}

.status-item .value {
    color: var(--primary-color);
}

/* 进度条样式 */
.progress-bar {
    width: 100%;
    height: 20px;
    background-color: var(--border-color);
    border-radius: 10px;
    overflow: hidden;
}

.progress {
    height: 100%;
    background-color: var(--primary-color);
    transition: width 0.3s;
}

/* 页脚样式 */
footer {
    background-color: var(--secondary-color);
    color: white;
    text-align: center;
    padding: 20px 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    nav {
        flex-direction: column;
    }
    
    nav ul {
        margin-top: 10px;
    }
    
    nav ul li {
        margin-left: 10px;
    }
}""")


def create_visualization_css(css_dir):
    """创建可视化CSS"""
    visualization_css = os.path.join(css_dir, 'visualization.css')
    if not os.path.exists(visualization_css):
        with open(visualization_css, 'w', encoding='utf-8') as f:
            f.write("""/* 可视化页面样式 */
.visualization-controls {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-group {
    display: flex;
    align-items: center;
}

.control-group label {
    margin-right: 10px;
    font-weight: bold;
}

.control-group input[type="range"] {
    width: 200px;
    margin-right: 10px;
}

.visualization-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-bottom: 30px;
}

.visualization-panel {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
}

.chart-container {
    height: 300px;
    position: relative;
}

.memory-container {
    height: 300px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 10px;
}

.memory-placeholder {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    color: #999;
    font-style: italic;
}

.memory-item {
    padding: 10px;
    margin-bottom: 10px;
    border-left: 3px solid var(--primary-color);
    background-color: #f9f9f9;
}

.memory-item.active {
    border-left-color: var(--accent-color);
    background-color: #fff5f5;
}

.memory-item .strength {
    float: right;
    font-weight: bold;
    color: var(--primary-color);
}

.memory-item.active .strength {
    color: var(--accent-color);
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

/* 响应式设计 */
@media (max-width: 992px) {
    .visualization-grid {
        grid-template-columns: 1fr;
    }
    
    .status-grid {
        grid-template-columns: 1fr;
    }
}""")


def create_network_css(css_dir):
    """创建网络CSS"""
    network_css = os.path.join(css_dir, 'network.css')
    if not os.path.exists(network_css):
        with open(network_css, 'w', encoding='utf-8') as f:
            f.write("""/* 神经网络页面样式 */
.network-controls {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-group {
    display: flex;
    align-items: center;
}

.control-group label {
    margin-right: 10px;
    font-weight: bold;
}

.control-group select {
    min-width: 150px;
}

.network-container {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 30px;
    height: 500px;
    position: relative;
}

#network-visualization {
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.network-details {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 30px;
}

.details-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
}

.detail-item {
    display: flex;
    flex-direction: column;
}

.detail-item .label {
    font-weight: bold;
    margin-bottom: 5px;
}

.detail-item .value {
    font-size: 1.2rem;
    color: var(--primary-color);
}

.selected-element-info {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
}

.info-container {
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 15px;
    min-height: 150px;
}

.info-placeholder {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    color: #999;
    font-style: italic;
}

/* 神经元和突触样式 */
.neuron {
    cursor: pointer;
    transition: fill 0.3s;
}

.neuron:hover {
    stroke-width: 2px;
}

.neuron.active {
    fill: var(--accent-color);
}

.synapse {
    stroke-width: 1px;
    transition: stroke-width 0.3s, stroke 0.3s;
}

.synapse.excitatory {
    stroke: #2ecc71;
}

.synapse.inhibitory {
    stroke: #e74c3c;
}

.synapse.active {
    stroke-width: 3px;
}

/* 响应式设计 */
@media (max-width: 992px) {
    .details-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 576px) {
    .details-grid {
        grid-template-columns: 1fr;
    }
}""")


def create_cognitive_css(css_dir):
    """创建认知CSS"""
    cognitive_css = os.path.join(css_dir, 'cognitive.css')
    if not os.path.exists(cognitive_css):
        with open(cognitive_css, 'w', encoding='utf-8') as f:
            f.write("""/* 认知过程页面样式 */
.cognitive-controls {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-group {
    display: flex;
    align-items: center;
}

.control-group label {
    margin-right: 10px;
    font-weight: bold;
}

.control-group select {
    min-width: 150px;
}

.cognitive-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-bottom: 30px;
}

.cognitive-panel {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
}

.chart-container {
    height: 250px;
    position: relative;
}

.interaction-container, .flow-container {
    height: 250px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
}

.cognitive-details {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 30px;
}

.details-content {
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 15px;
    min-height: 200px;
}

.details-placeholder {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    color: #999;
    font-style: italic;
}

/* 认知元素样式 */
.attention-focus {
    fill: var(--accent-color);
    opacity: 0.7;
    transition: opacity 0.3s, r 0.3s;
}

.attention-focus:hover {
    opacity: 1;
}

.memory-node {
    fill: var(--primary-color);
    stroke: white;
    stroke-width: 2px;
    transition: r 0.3s;
}

.memory-node.active {
    fill: var(--accent-color);
}

.memory-node:hover {
    r: 8;
}

.cognitive-link {
    stroke: #999;
    stroke-width: 1px;
    transition: stroke-width 0.3s;
}

.cognitive-link.active {
    stroke: var(--primary-color);
    stroke-width: 2px;
}

/* 响应式设计 */
@media (max-width: 992px) {
    .cognitive-grid {
        grid-template-columns: 1fr;
    }
}""")