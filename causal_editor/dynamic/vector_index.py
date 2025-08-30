"""动态向量索引
实时构建和查询向量索引，支持增量更新
"""

import torch
import numpy as np

try:
    import faiss
    import logging
    from typing import Dict, List, Tuple, Optional, Any
    from collections import defaultdict
    import time
    import threading
except ImportError as e:
    logging.warning(f'导入失败: {e}')
    # TODO: 添加fallback逻辑

logger = logging.getLogger(__name__)


class DynamicVectorIndex:
    """动态向量索引
    
    支持实时构建、增量更新和高效查询的向量索引系统
    """
    
    def __init__(
        self,
        dimension: int,
        device: str = "cuda",
        index_type: str = "flat",  # flat, hnsw, ivf
        max_vectors: int = 10000,
        similarity_threshold: float = 0.5
    ):
        """初始化动态向量索引
        
        Args:
            dimension: 向量维度
            device: 计算设备
            index_type: 索引类型 (flat, hnsw, ivf)
            max_vectors: 最大向量数量
            similarity_threshold: 相似度阈值
        """
        self.dimension = dimension
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.index_type = index_type
        self.max_vectors = max_vectors
        self.similarity_threshold = similarity_threshold
        
        # 索引管理
        self.indices = {}  # layer_id -> faiss_index
        self.metadata = {}  # layer_id -> [metadata_list]
        self.vector_counts = defaultdict(int)  # layer_id -> count
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 统计信息
        self.search_count = 0
        self.total_search_time = 0.0
        self.index_build_count = 0
        self.total_index_build_time = 0.0
        
        # 动态向量索引初始化完成
    
    def _create_index(self, layer_id: str) -> faiss.Index:
        """创建FAISS索引
        
        Args:
            layer_id: 层ID
            
        Returns:
            FAISS索引实例
        """
        if self.index_type == "flat":
            # 平坦索引，精确搜索
            index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
        elif self.index_type == "hnsw":
            # HNSW索引，近似搜索，速度快 - 也使用内积
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
        elif self.index_type == "ivf":
            # IVF索引，适合大规模数据
            nlist = min(100, max(1, self.max_vectors // 100))
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # 需要训练
            if not hasattr(self, '_ivf_trained'):
                # 生成随机训练数据
                train_data = np.random.random((max(1000, nlist * 10), self.dimension)).astype(np.float32)
                train_data = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)  # 归一化训练数据
                index.train(train_data)
                self._ivf_trained = True
        else:
            # 默认使用平坦索引
            index = faiss.IndexFlatIP(self.dimension)
        
        # 尝试转移到GPU
        if self.device.type == "cuda" and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                # 索引已转移到GPU
            except Exception as e:
                logging.warning(f"无法将索引转移到GPU: {e}")
        
        return index
    
    def add_vectors(
        self,
        layer_id: str,
        vectors: torch.Tensor,
        metadata_list: List[Dict[str, Any]]
    ):
        """添加向量到索引
        
        Args:
            layer_id: 层ID
            vectors: 向量张量 [num_vectors, dimension]
            metadata_list: 元数据列表
        """
        if vectors.size(0) != len(metadata_list):
            raise ValueError("向量数量与元数据数量不匹配")
        
        start_time = time.time()
        
        with self.lock:
            # 创建索引（如果不存在）
            if layer_id not in self.indices:
                self.indices[layer_id] = self._create_index(layer_id)
                self.metadata[layer_id] = []
            
            # 转换向量格式
            if isinstance(vectors, torch.Tensor):
                vectors_np = vectors.detach().cpu().numpy().astype(np.float32)
            else:
                vectors_np = np.array(vectors, dtype=np.float32)
            
            # 归一化向量（用于内积相似度）
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            vectors_np = vectors_np / norms
            
            # 检查容量限制
            current_count = self.vector_counts[layer_id]
            if current_count + vectors_np.shape[0] > self.max_vectors:
                # 需要清理旧数据或重建索引
                self._cleanup_index(layer_id)
            
            # 添加到索引
            try:
                self.indices[layer_id].add(vectors_np)
                self.metadata[layer_id].extend(metadata_list)
                self.vector_counts[layer_id] += vectors_np.shape[0]
                
                build_time = time.time() - start_time
                self.total_index_build_time += build_time
                self.index_build_count += 1
                
                # 向量添加完成
                
            except Exception as e:
                logging.error(f"添加向量到索引失败: {e}")
    
    def _cleanup_index(self, layer_id: str):
        """清理索引以释放空间
        
        Args:
            layer_id: 层ID
        """
        if layer_id in self.indices:
            # 简单策略：清空并重建索引
            self.indices[layer_id] = self._create_index(layer_id)
            self.metadata[layer_id] = []
            self.vector_counts[layer_id] = 0
            # 已清理索引
    
    def search(
        self,
        layer_id: str,
        query_vector: torch.Tensor,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量
        
        Args:
            layer_id: 层ID
            query_vector: 查询向量 [dimension]
            k: 返回的近邻数量
            score_threshold: 分数阈值
            
        Returns:
            搜索结果列表
        """
        if layer_id not in self.indices or self.vector_counts[layer_id] == 0:
            return []
        
        start_time = time.time()
        
        with self.lock:
            try:
                # 准备查询向量
                if isinstance(query_vector, torch.Tensor):
                    query_np = query_vector.detach().cpu().numpy().astype(np.float32)
                else:
                    query_np = np.array(query_vector, dtype=np.float32)
                
                # 确保是二维数组
                if query_np.ndim == 1:
                    query_np = query_np.reshape(1, -1)
                
                # 维度检查
                expected_dim = self.dimension
                actual_dim = query_np.shape[1]
                
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"查询向量维度不匹配: 期望 {expected_dim}, 实际 {actual_dim}. "
                        f"FAISS索引要求查询向量和索引向量维度完全一致。"
                        f"请检查query_vector的来源和预处理过程。"
                        f"当前query_vector形状: {query_np.shape}"
                    )
                
                # 归一化查询向量
                norm = np.linalg.norm(query_np)
                if norm > 0:
                    query_np = query_np / norm
                
                # 执行搜索
                scores, indices = self.indices[layer_id].search(query_np, min(k, self.vector_counts[layer_id]))
                
                # 处理结果
                results = []
                threshold = score_threshold if score_threshold is not None else self.similarity_threshold
                
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx >= 0 and score >= threshold:  # 有效索引且分数满足阈值
                        if idx < len(self.metadata[layer_id]):
                            result = self.metadata[layer_id][idx].copy()
                            result['similarity_score'] = float(score)
                            result['rank'] = i
                            results.append(result)
                
                search_time = time.time() - start_time
                self.total_search_time += search_time
                self.search_count += 1
                
                # logging.debug(f"在层 {layer_id} 搜索到 {len(results)} 个结果，耗时: {search_time*1000:.1f}ms")
                logger.debug(f"🔍 向量搜索: 层 {layer_id}, 找到 {len(results)} 个结果, 阈值 {threshold}")
                if results:
                    logger.debug(f"  📊 最高相似度: {results[0]['similarity_score']:.4f}")
                
                return results
                
            except Exception as e:
                logging.error(f"向量搜索失败: {e}")
                return []
    
    def search_all_layers(
        self,
        query_vector: torch.Tensor,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """在所有层中搜索
        
        Args:
            query_vector: 查询向量
            k: 每层返回的近邻数量
            score_threshold: 分数阈值
            
        Returns:
            各层搜索结果字典
        """
        results = {}
        
        with self.lock:
            for layer_id in self.indices:
                layer_results = self.search(layer_id, query_vector, k, score_threshold)
                if layer_results:
                    results[layer_id] = layer_results
        
        return results
    
    def get_layer_info(self, layer_id: str) -> Dict[str, Any]:
        """获取层信息
        
        Args:
            layer_id: 层ID
            
        Returns:
            层信息字典
        """
        with self.lock:
            if layer_id not in self.indices:
                return {}
            
            return {
                'layer_id': layer_id,
                'vector_count': self.vector_counts[layer_id],
                'index_type': self.index_type,
                'dimension': self.dimension,
                'has_gpu': hasattr(self.indices[layer_id], 'index')
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        with self.lock:
            total_vectors = sum(self.vector_counts.values())
            avg_search_time = self.total_search_time / max(self.search_count, 1)
            avg_build_time = self.total_index_build_time / max(self.index_build_count, 1)
            
            return {
                'total_vectors': total_vectors,
                'layer_count': len(self.indices),
                'search_count': self.search_count,
                'avg_search_time': avg_search_time,
                'total_search_time': self.total_search_time,
                'index_build_count': self.index_build_count,
                'avg_build_time': avg_build_time,
                'total_index_build_time': self.total_index_build_time,
                'layer_vector_counts': dict(self.vector_counts),
                'index_type': self.index_type,
                'dimension': self.dimension,
                'max_vectors': self.max_vectors
            }
    
    def clear_layer(self, layer_id: str):
        """清空指定层的索引
        
        Args:
            layer_id: 层ID
        """
        with self.lock:
            if layer_id in self.indices:
                del self.indices[layer_id]
                del self.metadata[layer_id]
                del self.vector_counts[layer_id]
                # 已清空层索引
    
    def clear_all(self):
        """清空所有索引"""
        with self.lock:
            self.indices.clear()
            self.metadata.clear()
            self.vector_counts.clear()
            # 已清空所有索引
    
    def __del__(self):
        """析构函数"""
        self.clear_all()