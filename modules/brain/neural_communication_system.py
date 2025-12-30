"""
神经通信系统

生产级的大脑区域间通信系统，提供高性能的消息传递、
事件处理和神经信号路由功能。
"""

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Callable, Any, Optional, Set
from uuid import uuid4


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageType(Enum):
    """消息类型"""
    NEURAL_SIGNAL = "neural_signal"
    MOTOR_COMMAND = "motor_command"
    SENSORY_INPUT = "sensory_input"
    COGNITIVE_STATE = "cognitive_state"
    SYSTEM_CONTROL = "system_control"


@dataclass
class NeuralMessage:
    """神经消息"""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    source_region: str = ""
    target_region: str = ""
    message_type: MessageType = MessageType.NEURAL_SIGNAL
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0  # 生存时间（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        return time.time() - self.timestamp > self.ttl


@dataclass
class MessageHandler:
    """消息处理器"""
    handler_id: str
    region: str
    callback: Callable[[NeuralMessage], None]
    message_types: Set[MessageType] = field(default_factory=set)
    active: bool = True


class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.routing_table: Dict[str, Set[str]] = defaultdict(set)
        self.region_priorities: Dict[str, int] = {}
        self.blocked_routes: Set[tuple] = set()
    
    def add_route(self, source: str, target: str, priority: int = 1) -> None:
        """添加路由"""
        self.routing_table[source].add(target)
        self.region_priorities[target] = max(
            self.region_priorities.get(target, 0), priority
        )
    
    def remove_route(self, source: str, target: str) -> None:
        """移除路由"""
        self.routing_table[source].discard(target)
    
    def block_route(self, source: str, target: str) -> None:
        """阻塞路由"""
        self.blocked_routes.add((source, target))
    
    def unblock_route(self, source: str, target: str) -> None:
        """解除路由阻塞"""
        self.blocked_routes.discard((source, target))
    
    def get_targets(self, source: str) -> Set[str]:
        """获取目标区域"""
        targets = self.routing_table.get(source, set())
        return {t for t in targets if (source, t) not in self.blocked_routes}
    
    def is_route_available(self, source: str, target: str) -> bool:
        """检查路由是否可用"""
        return (target in self.routing_table.get(source, set()) and 
                (source, target) not in self.blocked_routes)


class MessageQueue:
    """优先级消息队列"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {priority: deque() for priority in MessagePriority}
        self.lock = threading.RLock()
        self._size = 0
    
    def put(self, message: NeuralMessage) -> bool:
        """添加消息到队列"""
        with self.lock:
            if self._size >= self.max_size:
                # 移除最旧的低优先级消息
                self._remove_oldest_low_priority()
            
            if self._size < self.max_size:
                self.queues[message.priority].append(message)
                self._size += 1
                return True
            return False
    
    def get(self) -> Optional[NeuralMessage]:
        """从队列获取消息（按优先级）"""
        with self.lock:
            # 按优先级顺序检查队列
            for priority in sorted(MessagePriority, key=lambda x: x.value, reverse=True):
                queue = self.queues[priority]
                if queue:
                    message = queue.popleft()
                    self._size -= 1
                    
                    # 检查消息是否过期
                    if message.is_expired():
                        continue
                    
                    return message
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        return self._size
    
    def clear(self) -> None:
        """清空队列"""
        with self.lock:
            for queue in self.queues.values():
                queue.clear()
            self._size = 0
    
    def _remove_oldest_low_priority(self) -> None:
        """移除最旧的低优先级消息"""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL]:
            queue = self.queues[priority]
            if queue:
                queue.popleft()
                self._size -= 1
                return


class NeuralCommunicationSystem:
    """
    神经通信系统
    
    提供大脑区域间的高性能消息传递和事件处理功能。
    """
    
    def __init__(self, max_workers: int = 4, queue_size: int = 10000):
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.message_queue = MessageQueue(queue_size)
        self.router = MessageRouter()
        self.handlers: Dict[str, MessageHandler] = {}
        self.region_handlers: Dict[str, List[str]] = defaultdict(list)
        
        # 线程池和控制
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_processed': 0,
            'messages_dropped': 0,
            'handler_errors': 0,
            'routing_errors': 0
        }
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
    
    def start(self) -> None:
        """启动通信系统"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._message_processor, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("神经通信系统已启动")
    
    def stop(self) -> None:
        """停止通信系统"""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("神经通信系统已停止")
    
    def register_handler(self, region: str, callback: Callable[[NeuralMessage], None],
                        message_types: Optional[Set[MessageType]] = None) -> str:
        """
        注册消息处理器
        
        Args:
            region: 大脑区域名称
            callback: 消息处理回调函数
            message_types: 处理的消息类型集合
            
        Returns:
            处理器ID
        """
        handler_id = str(uuid4())
        
        handler = MessageHandler(
            handler_id=handler_id,
            region=region,
            callback=callback,
            message_types=message_types or set(MessageType)
        )
        
        self.handlers[handler_id] = handler
        self.region_handlers[region].append(handler_id)
        
        self.logger.info(f"注册消息处理器: {region} -> {handler_id}")
        return handler_id
    
    def unregister_handler(self, handler_id: str) -> bool:
        """
        注销消息处理器
        
        Args:
            handler_id: 处理器ID
            
        Returns:
            是否成功注销
        """
        if handler_id not in self.handlers:
            return False
        
        handler = self.handlers[handler_id]
        self.region_handlers[handler.region].remove(handler_id)
        del self.handlers[handler_id]
        
        self.logger.info(f"注销消息处理器: {handler_id}")
        return True
    
    def send_message(self, source_region: str, target_region: str,
                    payload: Any, message_type: MessageType = MessageType.NEURAL_SIGNAL,
                    priority: MessagePriority = MessagePriority.NORMAL,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        发送消息
        
        Args:
            source_region: 源区域
            target_region: 目标区域
            payload: 消息载荷
            message_type: 消息类型
            priority: 消息优先级
            metadata: 元数据
            
        Returns:
            是否成功发送
        """
        # 检查路由
        if not self.router.is_route_available(source_region, target_region):
            self.stats['routing_errors'] += 1
            self.logger.warning(f"路由不可用: {source_region} -> {target_region}")
            return False
        
        # 创建消息
        message = NeuralMessage(
            source_region=source_region,
            target_region=target_region,
            message_type=message_type,
            priority=priority,
            payload=payload,
            metadata=metadata or {}
        )
        
        # 添加到队列
        if self.message_queue.put(message):
            self.stats['messages_sent'] += 1
            self.logger.debug(f"消息已发送: {source_region} -> {target_region}")
            return True
        else:
            self.stats['messages_dropped'] += 1
            self.logger.warning(f"消息队列已满，消息被丢弃: {message.message_id}")
            return False
    
    def broadcast_message(self, source_region: str, payload: Any,
                         message_type: MessageType = MessageType.NEURAL_SIGNAL,
                         priority: MessagePriority = MessagePriority.NORMAL,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        广播消息到所有连接的区域
        
        Args:
            source_region: 源区域
            payload: 消息载荷
            message_type: 消息类型
            priority: 消息优先级
            metadata: 元数据
            
        Returns:
            成功发送的消息数量
        """
        targets = self.router.get_targets(source_region)
        sent_count = 0
        
        for target in targets:
            if self.send_message(source_region, target, payload, message_type, priority, metadata):
                sent_count += 1
        
        return sent_count
    
    def add_connection(self, source_region: str, target_region: str, priority: int = 1) -> None:
        """添加区域连接"""
        self.router.add_route(source_region, target_region, priority)
        self.logger.info(f"添加连接: {source_region} -> {target_region}")
    
    def remove_connection(self, source_region: str, target_region: str) -> None:
        """移除区域连接"""
        self.router.remove_route(source_region, target_region)
        self.logger.info(f"移除连接: {source_region} -> {target_region}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'queue_size': self.message_queue.size(),
            'active_handlers': len([h for h in self.handlers.values() if h.active]),
            'total_handlers': len(self.handlers),
            'performance_metrics': self.performance_monitor.get_metrics()
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
        self.performance_monitor.reset()
    
    def _message_processor(self) -> None:
        """消息处理器主循环"""
        while self.running:
            try:
                message = self.message_queue.get()
                if message is None:
                    time.sleep(0.001)  # 避免忙等待
                    continue
                
                self._process_message(message)
                
            except Exception as e:
                self.logger.error(f"消息处理错误: {e}")
                time.sleep(0.01)
    
    def _process_message(self, message: NeuralMessage) -> None:
        """处理单个消息"""
        start_time = time.perf_counter()
        
        try:
            # 获取目标区域的处理器
            handler_ids = self.region_handlers.get(message.target_region, [])
            
            if not handler_ids:
                self.logger.warning(f"未找到处理器: {message.target_region}")
                return
            
            # 并行处理消息
            futures = []
            for handler_id in handler_ids:
                handler = self.handlers.get(handler_id)
                if (handler and handler.active and 
                    message.message_type in handler.message_types):
                    
                    future = self.executor.submit(self._handle_message_safe, handler, message)
                    futures.append(future)
            
            # 等待所有处理器完成
            for future in futures:
                try:
                    future.result(timeout=1.0)  # 1秒超时
                except Exception as e:
                    self.stats['handler_errors'] += 1
                    self.logger.error(f"处理器执行错误: {e}")
            
            self.stats['messages_processed'] += 1
            
            # 记录性能指标
            processing_time = time.perf_counter() - start_time
            self.performance_monitor.record_processing_time(processing_time)
            
        except Exception as e:
            self.logger.error(f"消息处理失败: {e}")
    
    def _handle_message_safe(self, handler: MessageHandler, message: NeuralMessage) -> None:
        """安全地处理消息"""
        try:
            handler.callback(message)
        except Exception as e:
            self.logger.error(f"处理器 {handler.handler_id} 执行失败: {e}")
            raise


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_processing_time(self, processing_time: float) -> None:
        """记录处理时间"""
        with self.lock:
            self.processing_times.append(processing_time)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        with self.lock:
            if not self.processing_times:
                return {}
            
            times = list(self.processing_times)
            
            return {
                'avg_processing_time': sum(times) / len(times),
                'min_processing_time': min(times),
                'max_processing_time': max(times),
                'total_samples': len(times)
            }
    
    def reset(self) -> None:
        """重置监控数据"""
        with self.lock:
            self.processing_times.clear()


# 全局通信系统实例
_communication_system: Optional[NeuralCommunicationSystem] = None


def get_communication_system() -> NeuralCommunicationSystem:
    """获取全局通信系统实例"""
    global _communication_system
    if _communication_system is None:
        _communication_system = NeuralCommunicationSystem()
        _communication_system.start()
    return _communication_system


def publish_neural_event(source_region: str, target_region: str, payload: Any,
                        message_type: MessageType = MessageType.NEURAL_SIGNAL,
                        priority: MessagePriority = MessagePriority.NORMAL) -> bool:
    """便捷函数：发布神经事件"""
    system = get_communication_system()
    return system.send_message(source_region, target_region, payload, message_type, priority)


def subscribe_to_brain_region(region: str, callback: Callable[[NeuralMessage], None],
                             message_types: Optional[Set[MessageType]] = None) -> str:
    """便捷函数：订阅大脑区域消息"""
    system = get_communication_system()
    return system.register_handler(region, callback, message_types)


def reset_communication_system() -> None:
    """重置通信系统"""
    global _communication_system
    if _communication_system:
        _communication_system.stop()
        _communication_system = None