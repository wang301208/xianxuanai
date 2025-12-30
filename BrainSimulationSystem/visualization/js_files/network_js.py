"""
网络JS文件生成器
"""

import os


def create_network_js(js_dir):
    """创建网络JS"""
    network_js = os.path.join(js_dir, 'network.js')
    if not os.path.exists(network_js):
        with open(network_js, 'w', encoding='utf-8') as f:
            f.write("""// 神经网络JS文件
document.addEventListener('DOMContentLoaded', function() {
    console.log('神经网络页面已加载');
    
    // 获取元素
    const networkTypeSelect = document.getElementById('network-type');
    const layerCountInput = document.getElementById('layer-count');
    const neuronCountInput = document.getElementById('neuron-count');
    const activationFunctionSelect = document.getElementById('activation-function');
    const applyBtn = document.getElementById('apply-settings');
    const resetBtn = document.getElementById('reset-settings');
    
    // 初始化网络可视化
    let networkVisualization;
    initNetworkVisualization();
    
    // 绑定按钮事件
    applyBtn.addEventListener('click', applySettings);
    resetBtn.addEventListener('click', resetSettings);
    
    // 初始化网络可视化
    function initNetworkVisualization() {
        // 获取容器
        const container = document.getElementById('network-visualization');
        
        // 创建网络可视化
        networkVisualization = new NetworkVisualization(container);
        
        // 加载默认网络
        loadNetwork('feedforward');
    }
    
    // 应用设置
    function applySettings() {
        const networkType = networkTypeSelect.value;
        const layerCount = parseInt(layerCountInput.value);
        const neuronCount = parseInt(neuronCountInput.value);
        const activationFunction = activationFunctionSelect.value;
        
        // 发送设置到API
        fetch('/api/network/configure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                network_type: networkType,
                layer_count: layerCount,
                neuron_count: neuronCount,
                activation_function: activationFunction
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // 重新加载网络
                loadNetwork(networkType);
                
                // 显示成功消息
                showMessage('网络配置已更新', 'success');
            } else {
                // 显示错误消息
                showError('网络配置更新失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('网络配置更新失败:', error);
            showError('网络配置更新失败: ' + error.message);
        });
    }
    
    // 重置设置
    function resetSettings() {
        // 重置表单
        networkTypeSelect.value = 'feedforward';
        layerCountInput.value = '3';
        neuronCountInput.value = '10';
        activationFunctionSelect.value = 'relu';
        
        // 重新加载默认网络
        loadNetwork('feedforward');
        
        // 显示成功消息
        showMessage('网络设置已重置', 'success');
    }
    
    // 加载网络
    function loadNetwork(networkType) {
        fetch('/api/network/structure?type=' + networkType)
            .then(response => response.json())
            .then(data => {
                // 更新网络可视化
                networkVisualization.updateNetwork(data);
                
                // 更新网络详情
                updateNetworkDetails(data);
            })
            .catch(error => {
                console.error('加载网络结构失败:', error);
                showError('加载网络结构失败: ' + error.message);
            });
    }
    
    // 更新网络详情
    function updateNetworkDetails(data) {
        document.getElementById('neuron-count-value').textContent = formatNumber(data.neuron_count);
        document.getElementById('synapse-count-value').textContent = formatNumber(data.synapse_count);
        document.getElementById('layer-count-value').textContent = data.layer_count;
        document.getElementById('activation-function-value').textContent = data.activation_function;
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
});

// 网络可视化类
class NetworkVisualization {
    constructor(container) {
        this.container = container;
        this.width = container.clientWidth;
        this.height = container.clientHeight;
        this.neurons = [];
        this.synapses = [];
        this.selectedElement = null;
        
        // 创建SVG元素
        this.svg = d3.select(container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // 创建箭头标记
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 13)
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
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .on('tick', () => this.ticked());
        
        // 添加缩放功能
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.selectAll('g').attr('transform', event.transform);
            });
        
        this.svg.call(this.zoom);
        
        // 创建容器组
        this.container = this.svg.append('g');
        
        // 创建连接组
        this.linkGroup = this.container.append('g')
            .attr('class', 'links');
        
        // 创建节点组
        this.nodeGroup = this.container.append('g')
            .attr('class', 'nodes');
    }
    
    // 更新网络
    updateNetwork(data) {
        // 转换数据
        this.neurons = data.neurons.map(neuron => ({
            id: neuron.id,
            type: neuron.type,
            layer: neuron.layer,
            x: neuron.x || Math.random() * this.width,
            y: neuron.y || Math.random() * this.height,
            activation: neuron.activation || 0
        }));
        
        this.synapses = data.synapses.map(synapse => ({
            id: synapse.id,
            source: synapse.source,
            target: synapse.target,
            weight: synapse.weight,
            type: synapse.weight > 0 ? 'excitatory' : 'inhibitory'
        }));
        
        // 更新力导向图
        this.simulation.nodes(this.neurons);
        this.simulation.force('link').links(this.synapses);
        this.simulation.alpha(1).restart();
        
        // 更新视图
        this.updateView();
    }
    
    // 更新视图
    updateView() {
        // 更新连接
        const links = this.linkGroup.selectAll('.synapse')
            .data(this.synapses, d => d.id);
        
        links.exit().remove();
        
        const newLinks = links.enter()
            .append('line')
            .attr('class', d => 'synapse ' + d.type)
            .attr('marker-end', 'url(#arrowhead)')
            .on('click', (event, d) => this.selectElement(event, d));
        
        this.links = newLinks.merge(links);
        
        // 更新节点
        const nodes = this.nodeGroup.selectAll('.neuron')
            .data(this.neurons, d => d.id);
        
        nodes.exit().remove();
        
        const newNodes = nodes.enter()
            .append('circle')
            .attr('class', 'neuron')
            .attr('r', 8)
            .attr('fill', d => d.type === 'excitatory' ? '#3498db' : '#e74c3c')
            .call(d3.drag()
                .on('start', (event, d) => this.dragstarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragended(event, d)))
            .on('click', (event, d) => this.selectElement(event, d));
        
        this.nodes = newNodes.merge(nodes);
        
        // 添加标题
        this.nodes.append('title')
            .text(d => 'Neuron ' + d.id);
        
        this.links.append('title')
            .text(d => 'Synapse ' + d.id + '\\nWeight: ' + d.weight.toFixed(4));
    }
    
    // 力导向图更新
    ticked() {
        if (this.links) {
            this.links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }
        
        if (this.nodes) {
            this.nodes
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }
    }
    
    // 拖拽开始
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    // 拖拽中
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    // 拖拽结束
    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // 选择元素
    selectElement(event, d) {
        // 取消之前的选择
        if (this.selectedElement) {
            if (this.selectedElement.type) {
                // 神经元
                this.nodeGroup.selectAll('.neuron')
                    .filter(n => n.id === this.selectedElement.id)
                    .classed('active', false);
            } else {
                // 突触
                this.linkGroup.selectAll('.synapse')
                    .filter(s => s.id === this.selectedElement.id)
                    .classed('active', false);
            }
        }
        
        // 设置新的选择
        this.selectedElement = d;
        
        if (d.type) {
            // 神经元
            this.nodeGroup.selectAll('.neuron')
                .filter(n => n.id === d.id)
                .classed('active', true);
                
            // 更新神经元信息
            updateNeuronInfo(d);
        } else {
            // 突触
            this.linkGroup.selectAll('.synapse')
                .filter(s => s.id === d.id)
                .classed('active', true);
                
            // 更新突触信息
            updateSynapseInfo(d);
        }
    }
}

// 更新神经元信息
function updateNeuronInfo(neuron) {
    const infoContainer = document.getElementById('element-info');
    
    // 清空容器
    infoContainer.innerHTML = '';
    
    // 创建标题
    const title = document.createElement('h3');
    title.textContent = '神经元 #' + neuron.id;
    infoContainer.appendChild(title);
    
    // 创建信息表格
    const table = document.createElement('table');
    table.className = 'info-table';
    
    // 添加类型
    let row = table.insertRow();
    let cell1 = row.insertCell(0);
    let cell2 = row.insertCell(1);
    cell1.textContent = '类型';
    cell2.textContent = neuron.type === 'excitatory' ? '兴奋性' : '抑制性';
    
    // 添加层
    row = table.insertRow();
    cell1 = row.insertCell(0);
    cell2 = row.insertCell(1);
    cell1.textContent = '层';
    cell2.textContent = neuron.layer;
    
    // 添加激活值
    row = table.insertRow();
    cell1 = row.insertCell(0);
    cell2 = row.insertCell(1);
    cell1.textContent = '激活值';
    cell2.textContent = neuron.activation.toFixed(4);
    
    // 添加表格到容器
    infoContainer.appendChild(table);
    
    // 添加激活历史图表
    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    chartContainer.style.height = '100px';
    chartContainer.style.marginTop = '15px';
    infoContainer.appendChild(chartContainer);
    
    // 创建画布
    const canvas = document.createElement('canvas');
    chartContainer.appendChild(canvas);
    
    // 创建图表
    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: '激活值历史',
                data: Array(20).fill(0),
                borderColor: neuron.type === 'excitatory' ? '#3498db' : '#e74c3c',
                backgroundColor: neuron.type === 'excitatory' ? 'rgba(52, 152, 219, 0.1)' : 'rgba(231, 76, 60, 0.1)',
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
}

// 更新突触信息
function updateSynapseInfo(synapse) {
    const infoContainer = document.getElementById('element-info');
    
    // 清空容器
    infoContainer.innerHTML = '';
    
    // 创建标题
    const title = document.createElement('h3');
    title.textContent = '突触 #' + synapse.id;
    infoContainer.appendChild(title);
    
    // 创建信息表格
    const table = document.createElement('table');
    table.className = 'info-table';
    
    // 添加类型
    let row = table.insertRow();
    let cell1 = row.insertCell(0);
    let cell2 = row.insertCell(1);
    cell1.textContent = '类型';
    cell2.textContent = synapse.type === 'excitatory' ? '兴奋性' : '抑制性';
    
    // 添加源神经元
    row = table.insertRow();
    cell1 = row.insertCell(0);
    cell2 = row.insertCell(1);
    cell1.textContent = '源神经元';
    cell2.textContent = synapse.source.id;
    
    // 添加目标神经元
    row = table.insertRow();
    cell1 = row.insertCell(0);
    cell2 = row.insertCell(1);
    cell1.textContent = '目标神经元';
    cell2.textContent = synapse.target.id;
    
    // 添加权重
    row = table.insertRow();
    cell1 = row.insertCell(0);
    cell2 = row.insertCell(1);
    cell1.textContent = '权重';
    cell2.textContent = synapse.weight.toFixed(4);
    
    // 添加表格到容器
    infoContainer.appendChild(table);
    
    // 添加权重历史图表
    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    chartContainer.style.height = '100px';
    chartContainer.style.marginTop = '15px';
    infoContainer.appendChild(chartContainer);
    
    // 创建画布
    const canvas = document.createElement('canvas');
    chartContainer.appendChild(canvas);
    
    // 创建图表
    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: '权重历史',
                data: Array(20).fill(synapse.weight),
                borderColor: synapse.type === 'excitatory' ? '#2ecc71' : '#e74c3c',
                backgroundColor: synapse.type === 'excitatory' ? 'rgba(46, 204, 113, 0.1)' : 'rgba(231, 76, 60, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: -1,
                    max: 1
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}""")