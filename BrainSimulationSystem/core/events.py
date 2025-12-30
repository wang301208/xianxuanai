"""
事件总线系统

实现模块间松耦合通信
"""

from typing import Callable, Dict, List
import numpy as np

class EventBus:
    _subscriptions: Dict[str, List[Callable]] = {}
    
    @classmethod
    def subscribe(cls, event_type: str, callback: Callable):
        """订阅事件"""
        if event_type not in cls._subscriptions:
            cls._subscriptions[event_type] = []
        cls._subscriptions[event_type].append(callback)
        
    @classmethod 
    def publish(cls, event_type: str, **data):
        """发布事件"""
        if event_type in cls._subscriptions:
            for callback in cls._subscriptions[event_type]:
                callback(**data)