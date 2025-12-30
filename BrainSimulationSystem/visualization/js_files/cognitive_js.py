"""
认知JS文件生成器
"""

import os


def create_cognitive_js(js_dir):
    """创建认知JS"""
    cognitive_js = os.path.join(js_dir, 'cognitive.js')
    if not os.path.exists(cognitive_js):
        with open(cognitive_js, 'w', encoding='utf-8') as f:
            f.write("""// 认知过程JS文件
document.addEventListener('DOMContentLoaded', function() {
    console.log('认知过程页面已加载');
    
    // 获取元素
    const cognitiveTypeSelect = document.getElementById('cognitive-type');
    const applyBtn = document.getElementById('apply-settings');
    const resetBtn = document.getElementById('reset-settings');
    
    // 初始化认知可视化
    let cognitiveVisualization;
    initCognitiveVisualization();
    
    // 绑定按钮事件
    applyBtn.addEventListener('click', applySettings);
    resetBtn.addEventListener('click', resetSettings);
    
    // 初始化认知可视化
    function initCognitiveVisualization() {
        // 获取容器
        const interactionContainer = document.getElementById('interaction-visualization');
        const flowContainer = document.getElementById('flow-visualization');
        
        // 创建认知可视化
        cognitiveVisualization = new CognitiveVisualization(interactionContainer, flowContainer);
        
        // 加载默认认知过程
        loadCognitiveProcess('default');
    }
    
    // 应用设置
    function applySettings() {
        const cognitiveType = cognitiveTypeSelect.value;
        
        // 发送设置到API
        fetch('/api/cognitive/configure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cognitive_type: cognitiveType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // 重新加载认知过程
                loadCognitiveProcess(cognitiveType);
                
                // 显示成功消息
                showMessage('认知过程配置已更新', 'success');
            } else {
                // 显示错误消息
                showError('认知过程配置更新失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('认知过程配置更新失败:', error);
            showError('认知过程配置更新失败: ' + error.message);
        });
    }
    
    // 重置设置
    function resetSettings() {
        // 重置表单
        cognitiveTypeSelect.value = 'default';
        
        // 重新加载默认认知过程
        loadCognitiveProcess('default');
        
        // 显示成功消息
        showMessage('认知过程设置已重置', 'success');
    }
    
    // 加载认知过程
    function loadCognitiveProcess(cognitiveType) {
        fetch('/api/cognitive/structure?type=' + cognitiveType)
            .then(response => response.json())
            .then(data => {
                // 更新认知可视化
                cognitiveVisualization.updateCognitiveProcess(data);
                
                // 更新认知详情
                updateCognitiveDetails(data);
            })
            .catch(error => {
                console.error('加载认知过程结构失败:', error);
                showError('加载认知过程结构失败: ' + error.message);
            });
    }
    
    // 更新认知详情
    function updateCognitiveDetails(data) {
        const detailsContainer = document.getElementById('cognitive-details');
        
        // 清空容器
        detailsContainer.innerHTML = '';
        
        // 创建标题
        const title = document.createElement('h3');
        title.textContent = '认知过程详情';
        detailsContainer.appendChild(title);
        
        // 创建描述
        const description = document.createElement('p');
        description.textContent = data.description || '无描述';
        detailsContainer.appendChild(description);
        
        // 创建组件列表
        const componentTitle = document.createElement('h4');
        componentTitle.textContent = '认知组件';
        componentTitle.style.marginTop = '15px';
        detailsContainer.appendChild(componentTitle);
        
        const componentList = document.createElement('ul');
        componentList.className = 'component-list';
        
        if (data.components && data.components.length > 0) {
            data.components.forEach(component => {
                const item = document.createElement('li');
                item.className = 'component-item';
                
                const name = document.createElement('strong');
                name.textContent = component.name;
                
                const desc = document.createElement('span');
                desc.textContent = ' - ' + component.description;
                
                item.appendChild(name);
                item.appendChild(desc);
                componentList.appendChild(item);
            });
        } else {
            const item = document.createElement('li');
            item.textContent = '无组件';
            componentList.appendChild(item);
        }
        
        detailsContainer.appendChild(componentList);
        
        // 创建状态信息
        const stateTitle = document.createElement('h4');
        stateTitle.textContent = '当前状态';
        stateTitle.style.marginTop = '15px';
        detailsContainer.appendChild(stateTitle);
        
        const stateInfo = document.createElement('div');
        stateInfo.className = 'state-info';
        
        if (data.state) {
            const stateList = document.createElement('ul');
            
            Object.entries(data.state).forEach(([key, value]) => {
                const item = document.createElement('li');
                
                const keyElem = document.createElement('strong');
                keyElem.textContent = key + ': ';
                
                const valueElem = document.createElement('span');
                valueElem.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                
                item.appendChild(keyElem);
                item.appendChild(valueElem);
                stateList.appendChild(item);
            });
            
            stateInfo.appendChild(stateList);
        } else {
            stateInfo.textContent = '无状态信息';
        }
        
        detailsContainer.appendChild(stateInfo);
    }
    
    // 显示消息
    function showMessage(message, type) {
        // 检查是否已存在消息容器
        let messageContainer = document.getElementById('message-container');
        
        if (!messageContainer) {
            // 创建消息容器
            messageContainer = document.createElement('div');
            messageContainer.id = 'message-container';
            messageContainer.style.position = 'fixed';
            messageContainer.style.top = '20px';
            messageContainer.style.right = '20px';
            messageContainer.style.zIndex = '9999';
            document.body.appendChild(messageContainer);
        }
        
        // 创建消息元素
        const messageElement = document.createElement('div');
        messageElement.className = 'message ' + type;
        messageElement.style.padding = '10px 15px';
        messageElement.style.borderRadius = '4px';
        messageElement.style.marginBottom = '10px';
        messageElement.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
        messageElement.style.display = 'flex';
        messageElement.style.justifyContent = 'space-between';
        messageElement.style.alignItems = 'center';
        
        // 设置消息样式
        if (type === 'success') {
            messageElement.style.backgroundColor = '#2ecc71';
            messageElement.style.color = 'white';
        } else if (type === 'warning') {
            messageElement.style.backgroundColor = '#f39c12';
            messageElement.style.color = 'white';
        } else if (type === 'error') {
            messageElement.style.backgroundColor = '#e74c3c';
            messageElement.style.color = 'white';
        } else {
            messageElement.style.backgroundColor = '#3498db';
            messageElement.style.color = 'white';
        }
        
        // 添加消息文本
        const messageText = document.createElement('span');
        messageText.textContent = message;
        messageElement.appendChild(messageText);
        
        // 添加关闭按钮
        const closeButton = document.createElement('button');
        closeButton.textContent = '×';
        closeButton.style.background = 'none';
        closeButton.style.border = 'none';
        closeButton.style.color = 'white';
        closeButton.style.fontSize = '20px';
        closeButton.style.cursor = 'pointer';
        closeButton.style.marginLeft = '10px';
        closeButton.onclick = function() {
            messageContainer.removeChild(messageElement);
        };
        messageElement.appendChild(closeButton);
        
        // 添加消息到容器
        messageContainer.appendChild(messageElement);
        
        // 3秒后自动移除消息
        setTimeout(function() {
            if (messageElement.parentNode === messageContainer) {
                messageContainer.removeChild(messageElement);
            }
        }, 3000);
    }
    
    // 定期更新数据
    setInterval(function() {
        if (cognitiveVisualization) {
            updateCognitiveData();
        }
    }, 1000);
    
    // 更新认知数据
    function updateCognitiveData() {
        fetch('/api/cognitive/data')
            .then(response => response.json())
            .then(data => {
                // 更新认知可视化
                cognitiveVisualization.updateData(data);
                
                // 更新认知图表
                updateCognitiveCharts(data);
            })
            .catch(error => {
                console.error('获取认知数据失败:', error);
            });
    }
    
    // 更新认知图表
    function updateCognitiveCharts(data) {
        // 更新注意力图表
        updateAttentionChart(data.attention);
        
        // 更新工作记忆图表
        updateWorkingMemoryChart(data.working_memory);
    }
});

// 认知可视化类
class CognitiveVisualization {
    constructor(interactionContainer, flowContainer) {
        this.interactionContainer = interactionContainer;
        this.flowContainer = flowContainer;
        this.width = interactionContainer.clientWidth;
        this.height = interactionContainer.clientHeight;
        this.nodes = [];
        this.links = [];
        
        // 创建交互可视化
        this.createInteractionVisualization();
        
        // 创建流程可视化
        this.createFlowVisualization();
    }
    
    // 创建交互可视化
    createInteractionVisualization() {
        // 创建SVG元素
        this.interactionSvg = d3.select(this.interactionContainer)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // 创建箭头标记
        this.interactionSvg.append('defs').append('marker')
            .attr('id', 'interaction-arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('stroke', 'none');
        
        // 创建力导向图
        this.interactionSimulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .on('tick', () => this.interactionTicked());
        
        // 添加缩放功能
        this.interactionZoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.interactionSvg.selectAll('g').attr('transform', event.transform);
            });
        
        this.interactionSvg.call(this.interactionZoom);
        
        // 创建容器组
        this.interactionContainer = this.interactionSvg.append('g');
        
        // 创建连接组
        this.interactionLinkGroup = this.interactionContainer.append('g')
            .attr('class', 'links');
        
        // 创建节点组
        this.interactionNodeGroup = this.interactionContainer.append('g')
            .attr('class', 'nodes');
    }
    
    // 创建流程可视化
    createFlowVisualization() {
        // 创建SVG元素
        this.flowSvg = d3.select(this.flowContainer)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // 创建容器组
        this.flowContainer = this.flowSvg.append('g');
        
        // 创建流程组
        this.flowGroup = this.flowContainer.append('g')
            .attr('class', 'flow');
    }
    
    // 更新认知过程
    updateCognitiveProcess(data) {
        // 更新交互可视化
        this.updateInteractionVisualization(data);
        
        // 更新流程可视化
        this.updateFlowVisualization(data);
    }
    
    // 更新交互可视化
    updateInteractionVisualization(data) {
        // 转换数据
        this.nodes = data.nodes.map(node => ({
            id: node.id,
            name: node.name,
            type: node.type,
            x: node.x || Math.random() * this.width,
            y: node.y || Math.random() * this.height,
            activation: node.activation || 0
        }));
        
        this.links = data.links.map(link => ({
            id: link.id,
            source: link.source,
            target: link.target,
            type: link.type,
            strength: link.strength || 0
        }));
        
        // 更新力导向图
        this.interactionSimulation.nodes(this.nodes);
        this.interactionSimulation.force('link').links(this.links);
        this.interactionSimulation.alpha(1).restart();
        
        // 更新视图
        this.updateInteractionView();
    }
    
    // 更新交互视图
    updateInteractionView() {
        // 更新连接
        const links = this.interactionLinkGroup.selectAll('.cognitive-link')
            .data(this.links, d => d.id);
        
        links.exit().remove();
        
        const newLinks = links.enter()
            .append('line')
            .attr('class', 'cognitive-link')
            .attr('marker-end', 'url(#interaction-arrowhead)')
            .style('stroke-width', d => Math.max(1, d.strength * 3) + 'px');
        
        this.interactionLinks = newLinks.merge(links);
        
        // 更新节点
        const nodes = this.interactionNodeGroup.selectAll('.cognitive-node')
            .data(this.nodes, d => d.id);
        
        nodes.exit().remove();
        
        const newNodes = nodes.enter()
            .append('g')
            .attr('class', 'cognitive-node')
            .call(d3.drag()
                .on('start', (event, d) => this.interactionDragstarted(event, d))
                .on('drag', (event, d) => this.interactionDragged(event, d))
                .on('end', (event, d) => this.interactionDragended(event, d)));
        
        // 添加圆形
        newNodes.append('circle')
            .attr('r', 10)
            .attr('fill', d => this.getNodeColor(d.type))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
        
        // 添加文本
        newNodes.append('text')
            .attr('dy', 25)
            .attr('text-anchor', 'middle')
            .text(d => d.name)
            .style('font-size', '12px')
            .style('fill', '#333');
        
        this.interactionNodes = newNodes.merge(nodes);
        
        // 添加标题
        this.interactionNodes.selectAll('circle')
            .append('title')
            .text(d => d.name);
    }
    
    // 更新流程可视化
    updateFlowVisualization(data) {
        // 清空容器
        this.flowGroup.selectAll('*').remove();
        
        // 检查是否有流程数据
        if (!data.flow || data.flow.length === 0) {
            this.flowGroup.append('text')
                .attr('x', this.width / 2)
                .attr('y', this.height / 2)
                .attr('text-anchor', 'middle')
                .text('无流程数据')
                .style('font-size', '14px')
                .style('fill', '#999');
            return;
        }
        
        // 创建流程图
        const flowHeight = 50;
        const flowSpacing = 20;
        const flowWidth = this.width - 40;
        const startX = 20;
        const startY = 30;
        
        // 创建流程节点
        data.flow.forEach((step, index) => {
            // 创建节点组
            const stepGroup = this.flowGroup.append('g')
                .attr('transform', `translate(${startX}, ${startY + index * (flowHeight + flowSpacing)})`);
            
            // 创建节点矩形
            stepGroup.append('rect')
                .attr('width', flowWidth)
                .attr('height', flowHeight)
                .attr('rx', 5)
                .attr('ry', 5)
                .attr('fill', this.getNodeColor(step.type))
                .attr('opacity', 0.7)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            // 创建节点文本
            stepGroup.append('text')
                .attr('x', flowWidth / 2)
                .attr('y', flowHeight / 2)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .text(step.name)
                .style('font-size', '14px')
                .style('fill', '#fff')
                .style('font-weight', 'bold');
            
            // 创建连接线
            if (index < data.flow.length - 1) {
                this.flowGroup.append('path')
                    .attr('d', `M${startX + flowWidth / 2},${startY + flowHeight + index * (flowHeight + flowSpacing)} L${startX + flowWidth / 2},${startY + flowHeight + index * (flowHeight + flowSpacing) + flowSpacing}`)
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('marker-end', 'url(#interaction-arrowhead)');
            }
        });
    }
    
    // 更新数据
    updateData(data) {
        // 更新节点激活值
        if (data.node_activations) {
            this.interactionNodes.selectAll('circle')
                .attr('r', d => {
                    const activation = data.node_activations[d.id] || 0;
                    return 10 + activation * 10;
                })
                .attr('opacity', d => {
                    const activation = data.node_activations[d.id] || 0;
                    return 0.5 + activation * 0.5;
                });
        }
        
        // 更新连接强度
        if (data.link_strengths) {
            this.interactionLinks
                .style('stroke-width', d => {
                    const strength = data.link_strengths[d.id] || d.strength;
                    return Math.max(1, strength * 3) + 'px';
                })
                .style('stroke-opacity', d => {
                    const strength = data.link_strengths[d.id] || d.strength;
                    return 0.3 + strength * 0.7;
                });
        }
        
        // 更新流程激活
        if (data.flow_activations) {
            this.flowGroup.selectAll('rect')
                .attr('opacity', (d, i) => {
                    const activation = data.flow_activations[i] || 0;
                    return 0.3 + activation * 0.7;
                })
                .attr('stroke-width', (d, i) => {
                    const activation = data.flow_activations[i] || 0;
                    return activation > 0.5 ? 3 : 1;
                });
        }
    }
    
    // 获取节点颜色
    getNodeColor(type) {
        const colorMap = {
            'attention': '#e74c3c',
            'memory': '#3498db',
            'perception': '#2ecc71',
            'decision': '#f39c12',
            'emotion': '#9b59b6',
            'language': '#1abc9c',
            'motor': '#e67e22',
            'default': '#95a5a6'
        };
        
        return colorMap[type] || colorMap.default;
    }
    
    // 交互力导向图更新
    interactionTicked() {
        if (this.interactionLinks) {
            this.interactionLinks
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }
        
        if (this.interactionNodes) {
            this.interactionNodes
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        }
    }
    
    // 交互拖拽开始
    interactionDragstarted(event, d) {
        if (!event.active) this.interactionSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    // 交互拖拽中
    interactionDragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    // 交互拖拽结束
    interactionDragended(event, d) {
        if (!event.active) this.interactionSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// 注意力图表
let attentionChart;

// 工作记忆图表
let workingMemoryChart;

// 初始化图表
document.addEventListener('DOMContentLoaded', function() {
    // 初始化注意力图表
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
    
    // 初始化工作记忆图表
    const workingMemoryCtx = document.getElementById('working-memory-chart').getContext('2d');
    workingMemoryChart = new Chart(workingMemoryCtx, {
        type: 'bar',
        data: {
            labels: ['项目1', '项目2', '项目3', '项目4', '项目5', '项目6', '项目7'],
            datasets: [{
                label: '激活强度',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(155, 89, 182, 0.7)',
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(155, 89, 182, 0.7)',
                    'rgba(52, 152, 219, 0.7)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)',
                    'rgba(52, 152, 219, 1)'
                ],
                borderWidth: 1
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
});

// 更新注意力图表
function updateAttentionChart(data) {
    if (!attentionChart || !data) return;
    
    // 更新数据
    attentionChart.data.datasets[0].data = [
        data.visual || 0,
        data.auditory || 0,
        data.tactile || 0,
        data.internal || 0,
        data.memory || 0,
        data.decision || 0
    ];
    
    // 更新图表
    attentionChart.update();
}

// 更新工作记忆图表
function updateWorkingMemoryChart(data) {
    if (!workingMemoryChart || !data || !data.items) return;
    
    // 更新标签
    workingMemoryChart.data.labels = data.items.map(item => item.name || '项目');
    
    // 更新数据
    workingMemoryChart.data.datasets[0].data = data.items.map(item => item.activation || 0);
    
    // 更新图表
    workingMemoryChart.update();
}""")