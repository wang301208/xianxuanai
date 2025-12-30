"""
JavaScript生成器

为可视化服务器创建默认JavaScript文件。
"""

import os
from .js_files.main_js import create_main_js
from .js_files.visualization_js import create_visualization_js
from .js_files.network_js import create_network_js
from .js_files.cognitive_js import create_cognitive_js


def create_js_files(js_dir):
    """创建所有JavaScript文件"""
    # 创建主JS
    create_main_js(js_dir)
    
    # 创建可视化JS
    create_visualization_js(js_dir)
    
    # 创建网络JS
    create_network_js(js_dir)
    
    # 创建认知JS
    create_cognitive_js(js_dir)


