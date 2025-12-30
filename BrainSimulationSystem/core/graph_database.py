"""
图数据库连接管理器
Graph Database Connection Manager
"""

from typing import Dict, List, Any, Optional
import logging

try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    neo4j = None

class GraphDatabase:
    """图数据库连接管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
        self.logger = logging.getLogger("GraphDatabase")
        
        if NEO4J_AVAILABLE and config.get('enabled', False):
            self._connect()
    
    def _connect(self):
        """连接到Neo4j数据库"""
        
        try:
            uri = self.config.get('uri', 'bolt://localhost:7687')
            username = self.config.get('username', 'neo4j')
            password = self.config.get('password', 'password')
            
            self.driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
            self.logger.info("已连接到Neo4j数据库")
        except Exception as e:
            self.logger.error(f"连接Neo4j失败: {e}")
            self.driver = None
    
    def create_neuron_node(self, neuron_id: int, properties: Dict[str, Any]):
        """创建神经元节点"""
        
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            CREATE (n:Neuron {id: $neuron_id})
            SET n += $properties
            """
            session.run(query, neuron_id=neuron_id, properties=properties)
    
    def create_connection_edge(self, pre_id: int, post_id: int, properties: Dict[str, Any]):
        """创建连接边"""
        
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MATCH (pre:Neuron {id: $pre_id})
            MATCH (post:Neuron {id: $post_id})
            CREATE (pre)-[r:CONNECTS_TO]->(post)
            SET r += $properties
            """
            session.run(query, pre_id=pre_id, post_id=post_id, properties=properties)
    
    def query_connections(self, neuron_id: int, direction: str = 'out') -> List[Dict[str, Any]]:
        """查询神经元连接"""
        
        if not self.driver:
            return []
        
        if direction == 'out':
            query = """
            MATCH (n:Neuron {id: $neuron_id})-[r:CONNECTS_TO]->(target:Neuron)
            RETURN target.id as target_id, r as connection
            """
        else:
            query = """
            MATCH (source:Neuron)-[r:CONNECTS_TO]->(n:Neuron {id: $neuron_id})
            RETURN source.id as source_id, r as connection
            """
        
        with self.driver.session() as session:
            result = session.run(query, neuron_id=neuron_id)
            return [record.data() for record in result]
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()