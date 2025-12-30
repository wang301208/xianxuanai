"""
静态文件生成器

为可视化服务器创建默认CSS和JavaScript文件。
"""

import os
from .css_generator import create_css_files
from .js_generator import create_js_files


def create_default_static_files():
    """创建默认静态文件"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    # 确保静态文件目录存在
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # 创建CSS目录
    css_dir = os.path.join(static_dir, 'css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
    
    # 创建JS目录
    js_dir = os.path.join(static_dir, 'js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
    
    # 创建CSS文件
    create_css_files(css_dir)
    
    # 创建JS文件
    create_js_files(js_dir)


