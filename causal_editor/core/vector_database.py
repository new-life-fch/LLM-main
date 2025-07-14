"""
向量数据库组件
用于存储和检索预计算的因果知识激活指纹
"""

import torch
import numpy as np
import faiss
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class VectorDatabase:
    """
    向量数据库，基于FAISS实现高效的相似度搜索
    存储预计算的因果知识三元组的激活指纹
    """
    
    def __init__(self, db_path: str, device: str = "cuda"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库文件路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.db_path = Path(db_path)
        
        # FAISS索引
        self.index = None
        self.dimension = None
        
        # 元数据存储
        self.metadata = {}  # 存储每个向量对应的知识三元组信息
        self.layer_indices = {}  # 每层的索引映射
        
        # 统计信息
        self.total_vectors = 0
        self.search_count = 0
        
        # 加载已有数据库
        self.load_database()
        
        logging.info(f"向量数据库初始化完成: {self.total_vectors} 条记录")
    
    def load_database(self):
        """加载已存在的数据库"""
        if not self.db_path.exists():
            logging.warning(f"数据库文件不存在: {self.db_path}")
            return
        
        try:
            # 加载FAISS索引
            index_path = self.db_path / "faiss.index"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                self.dimension = self.index.d
                self.total_vectors = self.index.ntotal
                
                # 如果有GPU，尝试转移到GPU
                if self.device.type == "cuda" and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                        logging.info("FAISS索引已转移到GPU")
                    except Exception as e:
                        logging.warning(f"无法将FAISS索引转移到GPU: {e}")
            
            # 加载元数据
            metadata_path = self.db_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', {})
                    self.layer_indices = data.get('layer_indices', {})
            
            logging.info(f"数据库加载完成: {self.total_vectors} 条记录")
            
        except Exception as e:
            logging.error(f"加载数据库失败: {e}")
            self.index = None
            self.metadata = {}
            self.layer_indices = {}
    
    def save_database(self):
        """保存数据库到磁盘"""
        if self.index is None:
            logging.warning("没有可保存的索引")
            return
        
        try:
            # 创建目录
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引（确保在CPU上）
            index_to_save = self.index
            if hasattr(self.index, 'index'):  # GPU索引
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            
            index_path = self.db_path / "faiss.index"
            faiss.write_index(index_to_save, str(index_path))
            
            # 保存元数据
            metadata_path = self.db_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'layer_indices': self.layer_indices,
                    'dimension': self.dimension,
                    'total_vectors': self.total_vectors
                }, f)
            
            logging.info(f"数据库已保存到: {self.db_path}")
            
        except Exception as e:
            logging.error(f"保存数据库失败: {e}")
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        layer_id: str,
        knowledge_triplets: List[Dict[str, Any]]
    ):
        """
        添加向量到数据库
        
        Args:
            vectors: 激活向量 [num_vectors, dimension]
            layer_id: 层ID
            knowledge_triplets: 对应的知识三元组信息
        """
        if vectors.shape[0] != len(knowledge_triplets):
            raise ValueError("向量数量与知识三元组数量不匹配")
        
        # 初始化索引
        if self.index is None:
            self.dimension = vectors.shape[1]
            # 使用HNSW索引以平衡速度和精度
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        
        # 记录起始索引
        start_idx = self.total_vectors
        
        # 添加向量到索引
        self.index.add(vectors.astype(np.float32))
        
        # 更新元数据
        for i, triplet in enumerate(knowledge_triplets):
            vector_idx = start_idx + i
            self.metadata[vector_idx] = {
                'layer_id': layer_id,
                'subject': triplet['subject'],
                'relation': triplet['relation'], 
                'object': triplet['object'],
                'text': triplet.get('text', ''),
                'confidence': triplet.get('confidence', 1.0)
            }
        
        # 更新层级索引
        if layer_id not in self.layer_indices:
            self.layer_indices[layer_id] = []
        self.layer_indices[layer_id].extend(range(start_idx, start_idx + len(knowledge_triplets)))
        
        self.total_vectors += len(knowledge_triplets)
        logging.info(f"添加了 {len(knowledge_triplets)} 条记录到层 {layer_id}")
    
    def search(
        self, 
        query_vector: torch.Tensor, 
        layer_id: Optional[str] = None,
        k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量 [dimension]
            layer_id: 限制搜索的层ID，None表示搜索所有层
            k: 返回的近邻数量
            score_threshold: 相似度阈值
            
        Returns:
            搜索结果列表，包含向量信息和元数据
        """
        if self.index is None:
            return []
        
        self.search_count += 1
        
        # 准备查询向量
        if isinstance(query_vector, torch.Tensor):
            query_np = query_vector.detach().cpu().numpy().astype(np.float32)
        else:
            query_np = np.array(query_vector, dtype=np.float32)
        
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        try:
            # 执行搜索
            scores, indices = self.index.search(query_np, k)
            
            results = []
            for i in range(k):
                if indices[0][i] == -1:  # FAISS返回-1表示没有更多结果
                    break
                    
                idx = indices[0][i]
                score = float(scores[0][i])
                
                # 过滤低分结果
                if score < score_threshold:
                    continue
                
                # 获取元数据
                if idx in self.metadata:
                    metadata = self.metadata[idx].copy()
                    
                    # 如果指定了层级，检查是否匹配
                    if layer_id is not None and metadata['layer_id'] != layer_id:
                        continue
                    
                    metadata['similarity_score'] = score
                    metadata['vector_index'] = idx
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            logging.error(f"搜索失败: {e}")
            return []
    
    def get_triplets_by_subject(self, subject: str, layer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        根据主体获取相关三元组
        
        Args:
            subject: 主体实体
            layer_id: 限制的层ID
            
        Returns:
            相关三元组列表
        """
        results = []
        for idx, metadata in self.metadata.items():
            if metadata['subject'].lower() == subject.lower():
                if layer_id is None or metadata['layer_id'] == layer_id:
                    results.append(metadata)
        return results
    
    def get_triplets_by_relation(self, relation: str, layer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        根据关系获取相关三元组
        
        Args:
            relation: 关系类型
            layer_id: 限制的层ID
            
        Returns:
            相关三元组列表
        """
        results = []
        for idx, metadata in self.metadata.items():
            if relation.lower() in metadata['relation'].lower():
                if layer_id is None or metadata['layer_id'] == layer_id:
                    results.append(metadata)
        return results
    
    def get_size(self) -> int:
        """获取数据库大小"""
        return self.total_vectors
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        layer_stats = {}
        for layer_id, indices in self.layer_indices.items():
            layer_stats[layer_id] = len(indices)
        
        return {
            'total_vectors': self.total_vectors,
            'dimension': self.dimension,
            'search_count': self.search_count,
            'layer_statistics': layer_stats,
            'db_path': str(self.db_path)
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.search_count = 0
    
    def optimize_index(self):
        """优化索引性能"""
        if self.index is None:
            return
        
        try:
            # 调整HNSW参数
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = min(200, self.total_vectors // 10)
                logging.info(f"索引优化完成，efSearch设置为: {self.index.hnsw.efSearch}")
        except Exception as e:
            logging.warning(f"索引优化失败: {e}") 