"""
主JS文件生成器
"""

import os


def create_main_js(js_dir):
    """创建主JS"""
    main_js = os.path.join(js_dir, 'main.js')
    if not os.path.exists(main_js):
        with open(main_js, 'w', encoding='utf-8') as f:
            f.write("""// 主JS文件
document.addEventListener('DOMContentLoaded', function() {
    console.log('大脑模拟系统已加载');
    
    // 检查API状态
    checkApiStatus();
    
    // 设置全局错误处理
    setupErrorHandling();
});

// 检查API状态
function checkApiStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                throw new Error('API服务器未响应');
            }
            return response.json();
        })
        .then(data => {
            console.log('API状态:', data);
        })
        .catch(error => {
            console.error('API状态检查失败:', error);
            showError('API服务器连接失败，请检查服务器是否运行。');
        });
}

// 设置错误处理
function setupErrorHandling() {
    window.addEventListener('error', function(event) {
        console.error('全局错误:', event.error);
        showError('发生错误: ' + event.error.message);
    });
}

// 显示错误消息
function showError(message) {
    // 检查是否已存在错误消息容器
    let errorContainer = document.getElementById('error-container');
    
    if (!errorContainer) {
        // 创建错误消息容器
        errorContainer = document.createElement('div');
        errorContainer.id = 'error-container';
        errorContainer.style.position = 'fixed';
        errorContainer.style.top = '20px';
        errorContainer.style.right = '20px';
        errorContainer.style.zIndex = '9999';
        document.body.appendChild(errorContainer);
    }
    
    // 创建错误消息元素
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.style.backgroundColor = '#e74c3c';
    errorElement.style.color = 'white';
    errorElement.style.padding = '10px 15px';
    errorElement.style.borderRadius = '4px';
    errorElement.style.marginBottom = '10px';
    errorElement.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
    errorElement.style.display = 'flex';
    errorElement.style.justifyContent = 'space-between';
    errorElement.style.alignItems = 'center';
    
    // 添加错误消息文本
    const messageText = document.createElement('span');
    messageText.textContent = message;
    errorElement.appendChild(messageText);
    
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
        errorContainer.removeChild(errorElement);
    };
    errorElement.appendChild(closeButton);
    
    // 添加错误消息到容器
    errorContainer.appendChild(errorElement);
    
    // 5秒后自动移除错误消息
    setTimeout(function() {
        if (errorElement.parentNode === errorContainer) {
            errorContainer.removeChild(errorElement);
        }
    }, 5000);
}

// 格式化数字
function formatNumber(num) {
    return num.toLocaleString();
}

// 格式化日期时间
function formatDateTime(date) {
    return new Date(date).toLocaleString();
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}""")