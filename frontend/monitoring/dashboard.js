/**
 * AutoGPT 监控仪表板脚本
 * 
 * 本脚本负责 AutoGPT 系统的监控数据可视化，通过图表展示系统运行状态、
 * 性能瓶颈和蓝图版本分布等关键指标。
 * 
 * 功能特性:
 * - 成功率统计图表（饼图）
 * - 性能瓶颈分析图表（柱状图）
 * - 蓝图版本分布图表（柱状图）
 * 
 * 技术栈:
 * - Chart.js 用于图表渲染
 * - Fetch API 用于数据获取
 * - 异步编程模式
 */

/**
 * 异步获取 JSON 数据的工具函数
 * 
 * @param {string} url - 要请求的 API 端点 URL
 * @returns {Promise<Object>} 解析后的 JSON 数据
 * 
 * 功能:
 * - 发送 HTTP GET 请求
 * - 自动解析 JSON 响应
 * - 提供统一的数据获取接口
 */
async function fetchJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

/**
 * 渲染监控仪表板的主函数
 * 
 * 负责获取所有监控数据并创建相应的图表可视化。
 * 包含三个主要的监控维度：成功率、性能瓶颈和蓝图版本。
 * 
 * 图表类型:
 * - successChart: 成功/失败比例的环形图
 * - bottleneckChart: 性能瓶颈分析的柱状图
 * - blueprintChart: 蓝图版本分布的柱状图
 * 
 * 数据来源:
 * - /api/monitoring/success: 成功率统计
 * - /api/monitoring/bottlenecks: 性能瓶颈数据
 * - /api/monitoring/blueprint_versions: 蓝图版本信息
 */
async function render() {
  // 并行获取所有监控数据
  const success = await fetchJSON('/api/monitoring/success');
  const bottlenecks = await fetchJSON('/api/monitoring/bottlenecks');
  const versions = await fetchJSON('/api/monitoring/blueprint_versions');

  // 创建成功率环形图
  // 使用绿色表示成功，红色表示失败，直观显示系统健康状态
  new Chart(document.getElementById('successChart'), {
    type: 'doughnut',
    data: {
      labels: ['Success', 'Failure'],
      datasets: [{
        data: [success.successes, success.failures],
        backgroundColor: ['#4caf50', '#f44336']  // 绿色和红色
      }]
    }
  });

  // 创建性能瓶颈柱状图
  // 显示各个组件或操作的性能问题分布
  new Chart(document.getElementById('bottleneckChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(bottlenecks),
      datasets: [{
        data: Object.values(bottlenecks),
        backgroundColor: '#2196f3'  // 蓝色主题
      }]
    }
  });

  // 创建蓝图版本分布柱状图
  // 显示不同蓝图版本的使用情况和分布
  new Chart(document.getElementById('blueprintChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(versions),
      datasets: [{
        data: Object.values(versions),
        backgroundColor: '#9c27b0'  // 紫色主题
      }]
    }
  });
}

// 页面加载完成后立即渲染仪表板
render();
