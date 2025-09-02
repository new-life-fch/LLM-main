"""RAG检索器模块
基于Wikipedia语料库的检索增强生成系统

本模块实现了RAG（Retrieval-Augmented Generation）检索器，
支持从Wikipedia全量数据中检索相关文本段落，并与现有的
知识图谱系统协同工作。

主要特性：
- 基于FAISS的高效向量检索
- Wikipedia全量数据支持
- 与现有指纹构建系统集成
- 灵活的检索策略配置
"""

import logging
import os
import gc
import pickle
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import psutil
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    logging.warning(f'导入失败: {e}')
    # TODO: 添加fallback逻辑

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logging.warning("FAISS not available. RAG retrieval will be disabled.")
    FAISS_AVAILABLE = False

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    logging.warning("Datasets library not available. Wikipedia loading will be limited.")
    DATASETS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logging.warning("Requests library not available. Web search will be disabled.")
    REQUESTS_AVAILABLE = False

from ..utils.path_config import get_path_config, get_rag_paths


@dataclass
class RAGDocument:
    """RAG文档数据类"""
    id: str
    title: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


# 为了向后兼容，提供Document别名
Document = RAGDocument


@dataclass
class RAGRetrievalResult:
    """RAG检索结果数据类"""
    document: RAGDocument
    score: float
    rank: int


class RAGRetriever:
    """RAG检索器
    
    基于FAISS向量数据库的高效文档检索系统，支持Wikipedia全量数据检索。
    与现有的CausalEditor系统无缝集成，提供检索增强的知识获取能力。
    
    Attributes:
        model_name (str): 用于编码的预训练模型名称
        index_path (str): FAISS索引文件路径
        documents_path (str): 文档数据库路径
        max_seq_length (int): 最大序列长度
        batch_size (int): 批处理大小
        device (str): 计算设备
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        top_candidates: int = 500,
        min_score: float = 0.3,
        index_path: Optional[str] = None,
        documents_path: Optional[str] = None,
        wikipedia_dataset: str = "wikimedia/wikipedia",
        max_seq_length: int = 512,
        batch_size: int = 512,
        device: str = "cuda",
        embedding_dim: int = 1024,  # BGE-large-en的embedding维度
        cache_dir: Optional[str] = None,
        # HNSW索引配置参数
        hnsw_m: int = 64,
        hnsw_ef_construction: int = 400,
        hnsw_ef_search: int = 200,
        # 分片配置参数
        enable_sharding: bool = False,
        shard_size: int = 2000000,
        # Fallback机制配置参数
        enable_fallback: bool = True,
        fallback_threshold_high: float = 0.6,
        fallback_threshold_medium: float = 0.4,
        fallback_threshold_low: float = 0.2,
        fallback_cache_ttl: int = 864000,
        fallback_cache_path: Optional[str] = None,
        # 动态阈值配置参数
        enable_dynamic_threshold: bool = True,
        threshold_adjustment_interval: int = 50,
        min_threshold: float = 0.1,
        max_threshold: float = 0.6,
    ):
        """初始化RAG检索器
        
        Args:
            model_name: 用于文本编码的预训练模型（默认使用BGE-large-en-v1.5）
            index_path: FAISS索引存储路径（可选，使用统一路径配置）
            documents_path: 文档数据库路径（可选，使用统一路径配置）
            wikipedia_dataset: Wikipedia数据集名称
            max_seq_length: 最大文本序列长度
            batch_size: 编码批处理大小（BGE模型建议使用较小批次）
            device: 计算设备（cuda/cpu）
            embedding_dim: 嵌入向量维度（BGE-large-en为1024维）
            cache_dir: 缓存目录（可选，使用统一路径配置）
        """
        # 初始化路径配置
        self.path_config = get_path_config()
        rag_paths = get_rag_paths()
        
        self.model_name = model_name
        self.top_candidates = top_candidates
        self.min_score = min_score
        
        # 确保所有路径都是基于项目根目录的绝对路径
        if index_path:
            index_path = Path(index_path)
            if not index_path.is_absolute():
                index_path = self.path_config.project_root / index_path
            self.index_path = index_path
        else:
            self.index_path = Path(rag_paths["index_path"])
            
        if documents_path:
            documents_path = Path(documents_path)
            if not documents_path.is_absolute():
                documents_path = self.path_config.project_root / documents_path
            self.documents_path = documents_path
        else:
            self.documents_path = Path(rag_paths["documents_path"])
            
        if cache_dir:
            cache_dir = Path(cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = self.path_config.project_root / cache_dir
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path(rag_paths["cache_dir"])
            
        self.wikipedia_dataset = wikipedia_dataset
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.embedding_dim = embedding_dim
        
        # HNSW索引配置（针对~10M chunks优化）
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        
        # 动态调整HNSW参数以适应大规模数据
        self._optimize_hnsw_params()
        
        # Reranker配置
        self.use_reranker = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.reranker_type = None  # "bge"
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型和tokenizer
        self._init_model()
        
        # 初始化FAISS索引
        self.index = None
        self.is_index_loaded = False
        
        # 初始化文档和chunk相关属性
        self.doc_ids = []
        self.documents = {}  # 存储文档内容
        self.chunks = []     # 存储文档块
        self.chunk_to_doc = {}  # chunk到文档的映射
        self.chunk_metadata = []  # 存储chunk的详细metadata
        
        # 初始化文档数据库
        self._init_document_db()
        
        # 统计信息
        self.retrieval_count = 0
        self.total_documents = 0
        self.index_build_time = 0.0
        
        # 索引性能监控
        self.index_performance = {
            'build_time': 0.0,
            'search_times': [],
            'memory_usage': 0.0,
            'index_size_mb': 0.0
        }
        
        # 索引分片配置（用于超大规模数据）
        self.enable_sharding = enable_sharding
        self.shard_size = shard_size
        self.shards = []
        
        # Fallback机制配置（根据RAG方案）
        self.enable_fallback = enable_fallback
        self.fallback_threshold_high = fallback_threshold_high  
        self.fallback_threshold_medium = fallback_threshold_medium  
        self.fallback_threshold_low = fallback_threshold_low   
        self.fallback_cache_ttl = fallback_cache_ttl
        
        # 设置fallback缓存路径
        if fallback_cache_path:
            fallback_cache_path = Path(fallback_cache_path)
            if not fallback_cache_path.is_absolute():
                fallback_cache_path = self.path_config.project_root / fallback_cache_path
            self.fallback_cache_path = fallback_cache_path
        else:
            self.fallback_cache_path = self.cache_dir / "fallback_cache.db"
        
        # 初始化Fallback缓存数据库
        self._init_fallback_cache()
        
        # 相关性分数统计和阈值优化
        self.score_statistics = {
            'query_count': 0,
            'avg_scores': [],
            'top3_scores': [],
            'fallback_triggered': 0,
            'threshold_adjustments': 0
        }
        
        # 动态阈值配置
        self.enable_dynamic_threshold = enable_dynamic_threshold
        self.threshold_adjustment_interval = threshold_adjustment_interval  
        self.min_threshold = min_threshold  # 最小阈值
        self.max_threshold = max_threshold   # 最大阈值
        
        # Fallback触发标记
        self._last_fallback_triggered = False
        
        logging.info(f"RAG检索器初始化完成: {model_name}")
    
    def _optimize_hnsw_params(self):
        """根据预期数据规模动态优化HNSW参数
        
        针对~10M chunks的大规模检索进行参数优化：
        - 增加M值提升连接度和检索质量
        - 优化efConstruction平衡构建时间和质量
        - 调整efSearch在检索速度和精度间取得平衡
        """
        # 预估数据规模（chunks数量）
        estimated_chunks = 10_000_000  # 10M chunks目标
        
        # 根据数据规模动态调整M值
        # 对于大规模数据，增加M值可以提升检索质量
        if estimated_chunks >= 5_000_000:  # 5M+
            self.hnsw_m = max(self.hnsw_m, 96)  # 增加到96
            self.hnsw_ef_construction = max(self.hnsw_ef_construction, 800)  # 增加构建质量
            self.hnsw_ef_search = max(self.hnsw_ef_search, 400)  # 平衡检索速度和精度
        elif estimated_chunks >= 1_000_000:  # 1M+
            self.hnsw_m = max(self.hnsw_m, 80)
            self.hnsw_ef_construction = max(self.hnsw_ef_construction, 600)
            self.hnsw_ef_search = max(self.hnsw_ef_search, 300)
        
        # 内存优化：对于超大规模数据，适当降低efConstruction以节省内存
        available_memory_gb = psutil.virtual_memory().total / (1024**3)
        if available_memory_gb < 32 and estimated_chunks >= 5_000_000:
            # 内存不足时，适当降低efConstruction
            self.hnsw_ef_construction = min(self.hnsw_ef_construction, 600)
            logging.warning(f"内存限制({available_memory_gb:.1f}GB)，调整efConstruction为{self.hnsw_ef_construction}")
        
        logging.info(f"HNSW参数优化完成: M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}, efSearch={self.hnsw_ef_search}")
        logging.info(f"预期支持数据规模: {estimated_chunks:,} chunks")
    
    def _update_index_performance(self, index_size: int, build_time: float):
        """更新索引性能统计
        
        Args:
            index_size: 索引中向量数量
            build_time: 构建时间（秒）
        """
        try:
            # 更新构建时间
            self.index_performance['build_time'] = build_time
            
            # 计算内存使用（估算）
            # HNSW索引内存使用 ≈ 向量数 × 向量维度 × 4字节 × (1 + M/32)
            vector_memory = index_size * self.embedding_dim * 4 / (1024 * 1024)  # MB
            hnsw_overhead = vector_memory * (1 + self.hnsw_m / 32)
            self.index_performance['memory_usage'] = hnsw_overhead
            
            # 计算索引文件大小
            if self.index_path.exists():
                file_size_mb = self.index_path.stat().st_size / (1024 * 1024)
                self.index_performance['index_size_mb'] = file_size_mb
            
            # 计算构建速度（向量/秒）
            build_speed = index_size / build_time if build_time > 0 else 0
            logging.info(f"索引构建速度: {build_speed:.0f} 向量/秒")
            
        except Exception as e:
            logging.warning(f"更新索引性能统计失败: {e}")
    
    def set_reranker(self, reranker_model_name: str = "BAAI/bge-reranker-large", final_top_k: int = 3, batch_size: int = 256):
        """设置重排序器（根据RAG方案优化）
        
        Args:
            reranker_model_name: 重排序模型名称
            final_top_k: 重排序后保留的结果数量
            batch_size: 重排序批量大小
        """
        self.final_top_k = final_top_k
        self.reranker_batch_size = batch_size
        
        try:
            # 优先使用BGE reranker（性能更好）
            if "bge-reranker" in reranker_model_name:
                try:
                    from FlagEmbedding import FlagReranker
                    logging.info(f"正在加载BGE重排序器: {reranker_model_name}")
                    
                    # 使用优化配置加载BGE reranker
                    self.reranker_model = FlagReranker(
                        reranker_model_name, 
                        use_fp16=True,  # 使用半精度提升速度
                        device=self.device
                    )
                    self.reranker_type = "bge"
                    self.use_reranker = True
                    
                    # 为重排器初始化专用的tokenizer
                    try:
                        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                            reranker_model_name,
                            trust_remote_code=True
                        )
                        logging.info(f"重排序器tokenizer加载成功: {reranker_model_name}")
                    except Exception as e:
                        logging.warning(f"重排序器tokenizer加载失败: {e}，将使用编码器tokenizer")
                        self.reranker_tokenizer = None
                    
                    logging.info(f"BGE重排序器加载成功，重排序top-{final_top_k}")
                    return
                    
                except ImportError:
                    logging.warning("FlagEmbedding库不可用，尝试使用sentence-transformers")
                except Exception as e:
                    logging.warning(f"BGE重排序器加载失败: {e}，尝试降级方案")
            
            
        except ImportError:
            logging.error("重排序器相关库不可用，无法使用重排序器")
            self.use_reranker = False
        except Exception as e:
            logging.error(f"重排序器加载失败: {e}")
            self.use_reranker = False
    
    def _batch_rerank_bge(self, pairs: List[List[str]]) -> List[float]:
        """BGE reranker批量重排序
        
        Args:
            pairs: [query, document] 对列表
            
        Returns:
            重排序分数列表
        """
        try:
            # BGE reranker支持批量处理
            batch_size = getattr(self, 'reranker_batch_size', 32)
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.reranker_model.compute_score(batch_pairs)
                
                # 确保分数是列表格式
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores] if len(batch_pairs) == 1 else batch_scores.tolist()
                
                all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logging.error(f"BGE批量重排序失败: {e}")
            # 返回原始分数（基于相似度）
            return [0.5] * len(pairs)
    
    def _rerank_results(self, query: str, results: List[RAGRetrievalResult]) -> List[RAGRetrievalResult]:
        """对检索结果进行重排序（根据RAG方案优化）
        
        实现top-500到top-k的高效重排序：
        1. 批量处理提升性能
        2. 智能文本截断和组合
        3. 返回top-K结果
        
        Args:
            query: 查询文本
            results: 初步检索结果
            
        Returns:
            重排序后的top-K结果
        """
        if not self.use_reranker or not results:
            return results
        
        try:
            rerank_start_time = time.time()
            
            # 准备重排序数据
            pairs = []
            for result in results:
                # 智能组合标题和文本，优化重排序效果
                title = result.document.title or ""
                text = result.document.text or ""
                
                # 构建用于重排序的文档文本
                if title and text:
                    # 标题 + 文本前512个token（适合reranker输入长度）
                    doc_text = f"{title}. {text}"
                    doc_text = self._truncate_text_by_tokens(doc_text, 512, use_reranker_tokenizer=True)
                elif title:
                    doc_text = title
                else:
                    # 文本前512个token（使用重排序器tokenizer）
                    doc_text = self._truncate_text_by_tokens(text, 512, use_reranker_tokenizer=True)
                
                pairs.append([query, doc_text])
            
            # 根据reranker类型进行批量重排序
            if hasattr(self, 'reranker_type') and self.reranker_type == "bge":
                # BGE reranker批量处理
                rerank_scores = self._batch_rerank_bge(pairs)
            
            # 更新结果分数
            for i, result in enumerate(results):
                if i < len(rerank_scores):
                    result.score = float(rerank_scores[i])
            
            # 按新分数排序并取top-K
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            top_k_results = reranked_results[:self.final_top_k]
            
            # 更新排名
            for i, result in enumerate(top_k_results):
                result.rank = i
            
            rerank_time = time.time() - rerank_start_time
            logging.info(f"重排序完成: {len(results)} -> {len(top_k_results)} 结果，耗时: {rerank_time:.3f}秒")
            
            return top_k_results
            
        except Exception as e:
            logging.error(f"重排序失败: {e}")
            # 返回原始结果的前K个
            return results[:getattr(self, 'rerank_top_k', 10)]
     
    def _clear_memory(self):
        """清理内存"""
        try:
            # 清理Python垃圾回收
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logging.debug("内存清理完成")
            
        except Exception as e:
            logging.warning(f"内存清理失败: {e}")
     
    def _init_model(self):
        """初始化编码模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 启用FP16推理以提升速度
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16  # 启用FP16
            ).to(self.device)
            logging.info(f"启用FP16推理模式")
                
            self.model.eval()
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logging.info(f"成功加载编码模型: {self.model_name}")
            
        except Exception as e:
            logging.error(f"无法加载编码模型 {self.model_name}: {e}")
            raise
    
    def _init_document_db(self):
        """初始化文档数据库"""
        try:
            self.conn = sqlite3.connect(str(self.documents_path), check_same_thread=False)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_title ON documents(title)
            """)
            self.conn.commit()
            
            # 获取文档总数
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            self.total_documents = cursor.fetchone()[0]
            
            logging.info(f"文档数据库初始化完成，当前文档数: {self.total_documents}")
            
        except Exception as e:
            logging.error(f"文档数据库初始化失败: {e}")
            raise
    
    def _init_fallback_cache(self):
        """初始化Fallback缓存数据库"""
        try:
            self.fallback_conn = sqlite3.connect(str(self.fallback_cache_path), check_same_thread=False)
            self.fallback_conn.execute("""
                CREATE TABLE IF NOT EXISTS fallback_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,
                    source TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            self.fallback_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON fallback_cache(expires_at)
            """)
            self.fallback_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash ON fallback_cache(query_hash)
            """)
            self.fallback_conn.commit()
            
            # 清理过期缓存
            self._cleanup_fallback_cache()
            
            logging.info("Fallback缓存数据库初始化完成")
            
        except Exception as e:
            logging.error(f"Fallback缓存数据库初始化失败: {e}")
            # Fallback缓存失败不应该影响主要功能
            self.enable_fallback = False
    
    def _cleanup_fallback_cache(self):
        """清理过期的Fallback缓存"""
        try:
            current_time = datetime.now().isoformat()
            cursor = self.fallback_conn.execute(
                "DELETE FROM fallback_cache WHERE expires_at < ?",
                (current_time,)
            )
            deleted_count = cursor.rowcount
            self.fallback_conn.commit()
            
            if deleted_count > 0:
                logging.info(f"清理了 {deleted_count} 条过期的Fallback缓存")
                
        except Exception as e:
            logging.warning(f"清理Fallback缓存失败: {e}")
    
    def load_flashrag_corpus(self, corpus_path: Optional[str] = None, max_docs: Optional[int] = None):
        """加载FlashRAG语料库
        
        Args:
            corpus_path: flashrag_corpus.jsonl文件路径，如果为None则使用项目根目录下的文件
            max_docs: 最大加载文档数量，如果为None则加载全部文档
        """
        import json
        
        # 设置语料库文件路径
        if corpus_path is None:
            corpus_path = self.path_config.project_root / "wiki_data/flashrag_corpus.jsonl"
        else:
            corpus_path = Path(corpus_path)
            if not corpus_path.is_absolute():
                corpus_path = self.path_config.project_root / corpus_path
        
        if not corpus_path.exists():
            logging.error(f"FlashRAG语料库文件不存在: {corpus_path}")
            return False
        
        try:
            if max_docs is not None:
                logging.info(f"开始加载FlashRAG语料库: {corpus_path} (限制最大文档数: {max_docs})")
            else:
                logging.info(f"开始加载FlashRAG语料库: {corpus_path} (加载全部文档)")
            
            # 批量插入文档和chunks
            batch_size = 1024
            documents = []
            chunks_metadata = []
            processed_docs = 0
            total_chunks = 0
            
            # 逐行读取JSONL文件
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line.strip())
                        doc_id = item.get("id", f"doc_{line_num}")
                        contents = item.get("contents", "")
                        # 从contents中提取title（第一行）和text（剩余内容）
                        lines = contents.split('\n', 1)
                        title = lines[0] if lines else f"Document {doc_id}"
                        text = lines[1] if len(lines) > 1 else contents
                        
                        # 由于语料库已经清洗并按512token分割，直接使用
                        if not text.strip():
                            continue
                        
                        # 保存原始文档
                        doc_metadata = {
                            "original_length": len(text),
                            "chunk_count": 1,  # 每个条目就是一个chunk
                            "quality_score": 1.0,  # 已经清洗过的高质量数据
                            "source": "flashrag_corpus",
                            "subset": "all"
                        }
                        documents.append((doc_id, title, text, str(doc_metadata)))
                        
                        # 创建chunk metadata（每个条目就是一个chunk）
                        chunk_metadata = {
                            "original_doc_id": doc_id,
                            "chunk_id": 0,  # 每个条目只有一个chunk
                            "chunk_text": text,
                            "chunk_length": len(text),
                            "chunk_tokens": 512  # 根据说明，每条都是512token
                        }
                        chunks_metadata.append(chunk_metadata)
                        total_chunks += 1
                        
                        processed_docs += 1
                        
                        # 检查是否达到最大文档数量限制
                        if max_docs is not None and processed_docs >= max_docs:
                            logging.info(f"已达到最大文档数量限制: {max_docs}，停止加载")
                            break
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"解析第{line_num+1}行JSON失败: {e}")
                        continue
                    
                    # 批量插入
                    if len(documents) >= batch_size:
                        self._batch_insert_documents(documents)
                        documents = []
                        
                    if (line_num + 1) % 50000 == 0:
                        logging.info(f"已处理 {line_num + 1} 行，生成 {processed_docs} 篇有效文档，{total_chunks} 个chunks")
            
            # 插入剩余文档
            if documents:
                self._batch_insert_documents(documents)
            
            # 保存chunk metadata
            self.chunk_metadata = chunks_metadata
            logging.info(f"生成了 {len(chunks_metadata)} 个chunks，平均每篇文档 {len(chunks_metadata)/max(processed_docs, 1):.1f} 个chunks")
            
            # 更新文档总数
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            self.total_documents = cursor.fetchone()[0]
            
            logging.info(f"FlashRAG语料库加载完成，总文档数: {self.total_documents}")
            return True
            
        except Exception as e:
            logging.error(f"加载FlashRAG语料库失败: {e}")
            return False
    
    def download_wikipedia_corpus(self, subset: str = "20231101.en", download_path: str = "./data", 
                                 force_download: bool = False) -> bool:
        """下载Wikipedia语料库到指定位置
        
        Args:
            subset: Wikipedia数据集子集
            download_path: 下载路径
            force_download: 是否强制重新下载
            
        Returns:
            是否下载成功
        """
        if not DATASETS_AVAILABLE:
            logging.error("datasets库不可用，无法下载Wikipedia数据")
            return False
        
        try:
            download_path = Path(download_path)
            # 确保相对路径基于项目根目录
            if not download_path.is_absolute():
                download_path = self.path_config.project_root / download_path
            download_path.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"开始下载Wikipedia数据集: {subset} 到 {download_path}")
            
            # 检查是否已存在
            if not force_download:
                # 检查缓存目录是否已有数据
                cache_files = list(download_path.glob("**/*"))
                if cache_files:
                    logging.info(f"检测到已存在的数据文件，跳过下载。如需重新下载，请设置force_download=True")
                    return True
            
            from datasets import load_dataset
            
            # 下载数据集（仅下载，不处理）
            logging.info("正在下载Wikipedia数据集...")
            dataset = load_dataset(self.wikipedia_dataset, subset, split="train", cache_dir=str(download_path))
            
            # 获取数据集信息
            dataset_size = len(dataset)
            logging.info(f"Wikipedia数据集下载完成")
            logging.info(f"数据集大小: {dataset_size} 篇文档")
            logging.info(f"存储位置: {download_path}")
            
            # 显示存储空间使用情况
            total_size = sum(f.stat().st_size for f in download_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            logging.info(f"占用磁盘空间: {size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            logging.error(f"下载Wikipedia数据失败: {e}")
            return False
    
    def _sample_high_quality_articles(self, dataset, max_docs: int): # TODO:采样逻辑得改
        """采样高质量Wikipedia文章
        
        根据文章长度、链接数量等指标筛选高质量文章，
        优先选择内容丰富、结构完整的文章用于开发测试。
        
        Args:
            dataset: 原始Wikipedia数据集
            max_docs: 需要采样的文档数量
            
        Returns:
            采样后的数据集
        """
        try:
            logging.info(f"开始采样 {max_docs} 篇高质量Wikipedia文章...")
            
            # 计算文章质量分数
            article_scores = []
            
            for i, item in enumerate(dataset):
                title = item.get("title", "")
                text = item.get("text", "")
                
                # 质量评估指标
                text_length = len(text)
                title_length = len(title)
                
                # 跳过过短或过长的文章
                if text_length < 500 or text_length > 60000:
                    continue
                    
                # 跳过特殊页面（重定向、消歧义等）
                if any(keyword in title.lower() for keyword in 
                       ["disambiguation", "redirect", "list of", "category:"]):
                    continue
                
                # 计算质量分数
                # 1. 文本长度分数（适中长度得分更高）
                length_score = min(text_length / 5000, 1.0) * 0.4
                
                # 2. 链接密度分数（估算内部链接数量）
                link_count = text.count("[[") + text.count("]]") 
                link_score = min(link_count / 100, 1.0) * 0.3
                
                # 3. 段落结构分数
                paragraph_count = text.count("\n\n")
                structure_score = min(paragraph_count / 10, 1.0) * 0.2
                
                # 4. 标题质量分数
                title_score = min(title_length / 50, 1.0) * 0.1 if title_length > 5 else 0
                
                total_score = length_score + link_score + structure_score + title_score
                
                article_scores.append((i, total_score))
                
                # 限制评估数量以提高速度
                if len(article_scores) >= max_docs * 3:
                    break
            
            # 按分数排序并选择前max_docs篇
            article_scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, score in article_scores[:max_docs]]
            
            logging.info(f"从 {len(article_scores)} 篇候选文章中选择了 {len(selected_indices)} 篇高质量文章")
            logging.info(f"平均质量分数: {sum(score for _, score in article_scores[:max_docs]) / len(selected_indices):.3f}")
            
            return dataset.select(selected_indices)
            
        except Exception as e:
            logging.error(f"高质量文章采样失败: {e}")
            # 降级到简单采样
            return dataset.select(range(min(max_docs, len(dataset))))
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理
        
        根据RAG方案要求，清洗无效文章（stub、短文本），
        并为后续的BGE tokenizer切分做准备。
        """
        # 移除多余空白字符
        text = " ".join(text.split())
        
        # 移除Wikipedia特有的标记
        import re
        # 移除引用标记
        text = re.sub(r'\[\d+\]', '', text)
        # 移除分类标记
        text = re.sub(r'Category:[^\n]*', '', text)
        # 移除文件标记
        text = re.sub(r'File:[^\n]*', '', text)
        # 移除模板标记
        text = re.sub(r'{{[^}]*}}', '', text)
        
        # 清理多余的换行和空格
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_valid_article(self, text: str, title: str) -> bool:
        """检查文章是否有效
        
        根据RAG方案要求，过滤stub、短文本和无效文章
        
        Args:
            text: 文章文本
            title: 文章标题
            
        Returns:
            是否为有效文章
        """
        # 基本长度检查
        if len(text.strip()) < 100:  # 至少100字符
            return False
        
        # 检查是否为stub文章
        stub_indicators = [
            "stub", "disambiguation", "redirect", "may refer to",
            "is a list of", "category:", "template:"
        ]
        
        text_lower = text.lower()
        title_lower = title.lower()
        
        for indicator in stub_indicators:
            if indicator in text_lower or indicator in title_lower:
                return False
        
        # 检查文本质量（段落结构）
        sentences = text.split('.')
        if len(sentences) < 3:  # 至少3个句子
            return False
        
        # 检查是否包含足够的实质内容
        words = text.split()
        if len(words) < 50:  # 至少50个单词
            return False
        
        return True
    
    def _calculate_quality_score(self, text: str, title: str) -> float:
        """计算文章质量分数
        
        Args:
            text: 文章文本
            title: 文章标题
            
        Returns:
            质量分数 (0-1)
        """
        score = 0.0
        
        # 长度分数 (0.3权重)
        length_score = min(len(text) / 5000, 1.0)  # 5000字符为满分
        score += length_score * 0.3
        
        # 结构分数 (0.3权重)
        sentences = text.split('.')
        paragraphs = text.split('\n')
        structure_score = min((len(sentences) + len(paragraphs)) / 50, 1.0)
        score += structure_score * 0.3
        
        # 词汇丰富度 (0.2权重)
        words = set(text.lower().split())
        vocab_score = min(len(words) / 500, 1.0)  # 500个不同单词为满分
        score += vocab_score * 0.2
        
        # 标题质量 (0.2权重)
        title_score = min(len(title.split()) / 10, 1.0)  # 10个单词的标题为满分
        score += title_score * 0.2
        
        return min(score, 1.0)
    
    def _chunk_text(self, text: str, title: str = "") -> List[Tuple[str, int]]:
        """将文本切分为chunks
        
        根据RAG方案要求：用BGE tokenizer切分（512 tokens / stride 128）
        
        Args:
            text: 原始文本
            title: 文档标题
            
        Returns:
            文本块和token数量的元组列表 [(chunk_text, token_count), ...]
        """
        chunks = []
        
        # 添加标题前缀
        full_text = f"{title}. {text}" if title else text
        
        # 使用BGE tokenizer进行精确的token级别切分
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        chunk_size = 512  # 根据方案要求设置为512 tokens
        stride = 128      # 根据方案要求设置stride为128
        
        for i in range(0, len(tokens), chunk_size - stride):
            chunk_tokens = tokens[i:i + chunk_size]
            
            # 解码为文本
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # 过滤过短的chunk（至少50个字符）
            if len(chunk_text.strip()) > 50:
                # 返回文本和实际token数量的元组
                chunks.append((chunk_text.strip(), len(chunk_tokens)))
            
            # 如果已经到达文本末尾，停止切分
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    def _truncate_text_by_tokens(self, text: str, max_tokens: int = 512, use_reranker_tokenizer: bool = True) -> str:
        """基于token数量截断文本
        
        Args:
            text: 原始文本
            max_tokens: 最大token数量
            use_reranker_tokenizer: 是否优先使用重排序器的tokenizer
            
        Returns:
            截断后的文本
        """
        if not text:
            return text
            
        # 选择合适的tokenizer
        tokenizer_to_use = self.reranker_tokenizer
           
        try:
            # 使用__call__方法进行tokenization（推荐方式，性能更好）
            tokenized = tokenizer_to_use(
                text,
                max_length=max_tokens,
                truncation=True,
                padding=False,  # 单个文本不需要padding
                add_special_tokens=False,
                return_tensors=None  # 返回Python列表而不是tensor
            )
            
            # 获取token ids
            tokens = tokenized['input_ids']
            
            # 如果token数量已经在限制内，直接返回原文本
            if len(tokens) <= max_tokens:
                return text
            
            # 解码截断后的tokens
            truncated_text = tokenizer_to_use.decode(tokens, skip_special_tokens=True)
            
            return truncated_text.strip()
            
        except Exception as e:
            logging.warning(f"Token截断失败，回退到字符截断: {e}")
            # 回退到字符截断（保守估计，1个token约等于4个字符）
            char_limit = max_tokens * 4
            return text[:char_limit]
    
    def _batch_insert_documents(self, documents: List[Tuple[str, str, str, str]]):
        """批量插入文档"""
        try:
            self.conn.executemany(
                "INSERT OR REPLACE INTO documents (id, title, text, metadata) VALUES (?, ?, ?, ?)",
                documents
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"批量插入文档失败: {e}")
    
    def build_index(self, rebuild: bool = False) -> bool:
        """构建FAISS索引
        
        Args:
            rebuild: 是否重新构建索引
            
        Returns:
            是否成功构建索引
        """
        if not FAISS_AVAILABLE:
            logging.error("FAISS不可用，无法构建索引")
            return False
        
        # 确保faiss模块可用
        import faiss
        
        # 检查是否已存在索引
        if not rebuild and self.index_path.exists():
            return self.load_index()
        
        start_time = time.time()
        
        try:
            logging.info("开始构建FAISS IVF+PQ索引...")
            
            # 检查是否需要启用分片
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            estimated_chunks = total_docs  # 估算每文档平均5个chunks
            
            if estimated_chunks > 5_000_000:
                self.enable_sharding = True
                logging.info(f"检测到大规模数据({estimated_chunks:,} chunks)，启用索引分片")
            
            # 创建FAISS IVF+PQ索引（根据用户要求）
            # 使用IVF+PQ索引，适合大规模数据检索
            # 动态调整nlist，确保训练数据足够
            nlist = min(65536, estimated_chunks // 10)  # 确保至少有10倍于聚类中心的训练数据
            nlist = max(nlist, 100)  # 最小值为100
            # 确保m是embedding_dim的因子
            if self.embedding_dim % 64 == 0:
                m = 64  # PQ编码的子向量数量
            elif self.embedding_dim % 32 == 0:
                m = 32
            else:
                m = 16  # 默认值
            nbits = 8  # 每个子向量的位数
            
            # 创建量化器
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            
            # 创建IVF+PQ索引
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
            
            # 尝试使用GPU加速
            if torch.cuda.is_available():
                try:
                    import faiss.contrib.torch_utils
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logging.info("成功启用FAISS GPU加速")
                except Exception as e:
                    logging.warning(f"无法启用FAISS GPU加速: {e}，使用CPU版本")
            
            logging.info(f"IVF+PQ索引配置: nlist={nlist}, m={m}, nbits={nbits}")
            logging.info(f"预期处理数据量: {estimated_chunks:,} chunks")
            
            # 检查是否已有chunk metadata（避免重复切分）
            if hasattr(self, 'chunk_metadata') and self.chunk_metadata:
                logging.info(f"使用已有的chunk metadata，包含 {len(self.chunk_metadata)} 个chunks")
                chunk_metadata = self.chunk_metadata
                
                embeddings = []
                doc_ids = []
                batch_texts = []
                chunk_count = len(chunk_metadata)
                
                # 直接使用已有的chunk数据
                for chunk_meta in chunk_metadata:
                    chunk_text = chunk_meta['chunk_text']
                    original_doc_id = chunk_meta['original_doc_id']
                    chunk_idx = chunk_meta['chunk_id']
                    
                    chunk_id = f"{original_doc_id}_chunk_{chunk_idx}"
                    batch_texts.append(chunk_text)
                    doc_ids.append(chunk_id)
                    
                    # 批量编码 - 使用合理的批次大小避免内存问题
                    if len(batch_texts) >= 10000:
                        batch_embeddings = self._encode_texts(batch_texts)
                        embeddings.extend(batch_embeddings)
                        batch_texts = []
                        
                        if len(embeddings) % 500 == 0:
                            logging.info(f"已编码 {len(embeddings)} 个chunks，进度: {len(embeddings)/len(chunk_metadata)*100:.1f}%")

            # 处理剩余文本
            if batch_texts:
                batch_embeddings = self._encode_texts(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # 添加到FAISS索引
            if embeddings:
                embeddings_array = np.array(embeddings).astype(np.float32)
                # L2归一化用于余弦相似度
                faiss.normalize_L2(embeddings_array)
                
                # 训练IVF索引（IVF索引需要先训练）
                logging.info("开始训练IVF索引...")
                train_size = min(len(embeddings_array), max(65536 * 10, 100000))  # 训练样本数量
                train_data = embeddings_array[:train_size]
                self.index.train(train_data)
                logging.info(f"IVF索引训练完成，使用 {train_size} 个样本")
                
                # 添加所有向量到索引
                logging.info("开始添加向量到索引...")
                self.index.add(embeddings_array)
                logging.info(f"成功添加 {len(embeddings_array)} 个向量到索引")
                
                # 保存索引
                faiss.write_index(self.index, str(self.index_path))
                
                # 保存文档ID映射
                id_mapping_path = self.index_path.with_suffix(".ids")
                with open(id_mapping_path, "wb") as f:
                    pickle.dump(doc_ids, f)
                
                # 保存chunk metadata
                metadata_path = self.index_path.with_suffix(".metadata")
                with open(metadata_path, "wb") as f:
                    pickle.dump(chunk_metadata, f)
                
                # 设置实例变量
                self.doc_ids = doc_ids
                self.chunk_metadata = chunk_metadata
                self.is_index_loaded = True
                self.index_build_time = time.time() - start_time
                
                # 更新性能统计
                self._update_index_performance(len(embeddings), self.index_build_time)
                
                logging.info(f"FAISS索引构建完成，耗时: {self.index_build_time:.2f}秒")
                logging.info(f"索引大小: {len(embeddings)} 个向量")
                logging.info(f"总chunks数: {chunk_count}")
                logging.info(f"平均每文档chunks数: {chunk_count / len(set(meta['original_doc_id'] for meta in chunk_metadata)):.1f}")
                logging.info(f"索引内存使用: {self.index_performance['memory_usage']:.1f}MB")
                logging.info(f"索引文件大小: {self.index_performance['index_size_mb']:.1f}MB")
                
                return True
            else:
                logging.error("没有可用的文档进行索引构建")
                return False
                
        except Exception as e:
            logging.error(f"构建FAISS索引失败: {e}")
            return False
    
    def load_index(self) -> bool:
        """加载已存在的FAISS索引"""
        if not FAISS_AVAILABLE:
            return False
        
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                
                # 加载文档ID映射
                id_mapping_path = self.index_path.with_suffix(".ids")
                if id_mapping_path.exists():
                    with open(id_mapping_path, "rb") as f:
                        self.doc_ids = pickle.load(f)
                else:
                    logging.warning("文档ID映射文件不存在")
                    return False
                
                # 加载chunk metadata
                metadata_path = self.index_path.with_suffix(".metadata")
                if metadata_path.exists():
                    with open(metadata_path, "rb") as f:
                        self.chunk_metadata = pickle.load(f)
                    logging.info(f"成功加载chunk metadata，包含 {len(self.chunk_metadata)} 个chunks")
                else:
                    logging.warning("Chunk metadata文件不存在，使用默认metadata")
                    self.chunk_metadata = []
                
                self.is_index_loaded = True
                logging.info(f"成功加载FAISS索引，包含 {self.index.ntotal} 个向量")
                return True
            else:
                logging.warning("FAISS索引文件不存在")
                return False
                
        except Exception as e:
            logging.error(f"加载FAISS索引失败: {e}")
            return False
    
    def _encode_texts(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        """批量编码文本（优化版本：预分词 + DataLoader）
        
        Args:
            texts: 待编码的文本列表
            is_query: 是否为查询文本（BGE模型对查询和文档使用不同的处理方式）
        """
        if not texts:
            return []
        
        try:
            # BGE模型的特殊处理：为查询添加指令前缀
            if "bge" in self.model_name.lower() and is_query:
                processed_texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            else:
                processed_texts = texts
            
            # 预分词优化：批量tokenize到CPU内存
            logging.info(f"开始预分词 {len(processed_texts)} 个文本...")
            tokenized_inputs = self.tokenizer(
                processed_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # 创建数据集和DataLoader
            dataset = TensorDataset(
                tokenized_inputs['input_ids'],
                tokenized_inputs['attention_mask']
            )
            
            # 使用DataLoader进行批量处理，启用多进程和内存固定
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=min(10, len(processed_texts) // self.batch_size + 1),  # 动态调整worker数量
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False
            )
            
            embeddings = []
            total_batches = len(dataloader)
            
            logging.info(f"开始编码，共 {total_batches} 个批次...")
            
            for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
                try:
                    # 将数据移动到GPU（如果使用pin_memory，这会更快）
                    inputs = {
                        'input_ids': input_ids.to(self.device, non_blocking=True),
                        'attention_mask': attention_mask.to(self.device, non_blocking=True)
                    }
                    
                    # 编码
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                        # BGE模型官方推荐使用CLS池化方法
                        if "bge" in self.model_name.lower():
                            # BGE模型使用CLS token池化（推荐方法）
                            batch_embeddings = outputs.last_hidden_state[:, 0]  # 取[CLS] token
                        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            # 其他模型使用pooler_output
                            batch_embeddings = outputs.pooler_output
                        else:
                            # 降级方案：平均池化
                            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                        
                        # 转换为numpy并添加到结果列表
                        batch_embeddings_np = batch_embeddings.cpu().numpy()
                        embeddings.extend([emb for emb in batch_embeddings_np])
                        
                        # 清理GPU内存
                        del inputs, outputs, batch_embeddings
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 显示进度
                        if len(processed_texts) > 100:
                            # 确保显示第一个批次和每10%的进度
                            progress_interval = max(1, total_batches // 10)
                            if batch_idx == 0 or (batch_idx + 1) % progress_interval == 0 or batch_idx + 1 == total_batches:
                                progress = (batch_idx + 1) / total_batches * 100
                                logging.info(f"编码进度: {progress:.1f}% ({batch_idx + 1}/{total_batches} 批次)")
                        
                        # 定期清理内存
                        if (batch_idx + 1) % 10 == 0 or len(embeddings) > 1000:
                            self._clear_memory()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logging.error(f"GPU内存不足，当前批次大小: {self.batch_size}")
                        logging.error("建议减少batch_size或使用更小的模型")
                        self._clear_memory()
                        raise e
                    else:
                        raise e
            
            # 最终清理内存
            self._clear_memory()
            
            logging.info(f"编码完成，共生成 {len(embeddings)} 个向量")
            return embeddings
            
        except Exception as e:
            logging.error(f"文本编码失败: {e}")
            return []
    
    def _update_score_statistics(self, query: str, results: List[RAGRetrievalResult]):
        """更新分数统计和阈值优化
        
        Args:
            query: 查询文本
            results: 检索结果
        """
        try:
            if not results:
                return
            
            # 更新查询计数
            self.score_statistics['query_count'] += 1
            
            # 计算平均分数
            avg_score = sum(r.score for r in results) / len(results)
            self.score_statistics['avg_scores'].append(avg_score)
            
            # 计算top-3分数
            top3_results = results[:3]
            
            if top3_results:
                top3_avg = sum(r.score for r in top3_results) / len(top3_results)
                self.score_statistics['top3_scores'].append(top3_avg)
            
            # 检查是否触发了Fallback
            if hasattr(self, '_last_fallback_triggered') and self._last_fallback_triggered:
                self.score_statistics['fallback_triggered'] += 1
                self._last_fallback_triggered = False
            
            # 动态阈值优化
            if (self.enable_dynamic_threshold and 
                self.score_statistics['query_count'] % self.threshold_adjustment_interval == 0):
                self._optimize_thresholds()
            
            # 记录统计信息（每100次查询）
            if self.score_statistics['query_count'] % 100 == 0:
                self._log_score_statistics()
                
        except Exception as e:
            logging.warning(f"更新分数统计失败: {e}")
    
    def _optimize_thresholds(self):
        """动态优化阈值
        
        根据历史分数统计调整Fallback阈值，提升检索效果
        """
        try:
            if len(self.score_statistics['top3_scores']) < 50:  # 至少需要50个样本
                return
            
            # 计算最近100次查询的统计信息
            recent_scores = self.score_statistics['top3_scores'][-100:]
            recent_avg = sum(recent_scores) / len(recent_scores)
            recent_std = np.std(recent_scores)
            
            # 计算Fallback触发率
            recent_queries = min(100, self.score_statistics['query_count'])
            fallback_rate = self.score_statistics['fallback_triggered'] / recent_queries
            
            # 阈值优化策略
            old_high = self.fallback_threshold_high
            old_medium = self.fallback_threshold_medium
            old_low = self.fallback_threshold_low
            
            # 如果Fallback触发率过高（>50%），降低阈值
            if fallback_rate > 0.5:
                self.fallback_threshold_high = max(self.min_threshold, self.fallback_threshold_high - 0.05)
                self.fallback_threshold_medium = max(self.min_threshold, self.fallback_threshold_medium - 0.03)
                self.fallback_threshold_low = max(self.min_threshold, self.fallback_threshold_low - 0.02)
            
            # 如果Fallback触发率过低（<10%），提高阈值
            elif fallback_rate < 0.1:
                self.fallback_threshold_high = min(self.max_threshold, self.fallback_threshold_high + 0.03)
                self.fallback_threshold_medium = min(self.max_threshold, self.fallback_threshold_medium + 0.02)
                self.fallback_threshold_low = min(self.max_threshold, self.fallback_threshold_low + 0.01)
            
            # 基于分数分布调整
            if recent_avg > 0.4:  # 整体分数较高
                self.fallback_threshold_high = min(self.max_threshold, recent_avg + recent_std)
                self.fallback_threshold_medium = min(self.max_threshold, recent_avg)
                self.fallback_threshold_low = min(self.max_threshold, recent_avg - recent_std)
            
            # 记录阈值调整
            if (old_high != self.fallback_threshold_high or 
                old_medium != self.fallback_threshold_medium or 
                old_low != self.fallback_threshold_low):
                
                self.score_statistics['threshold_adjustments'] += 1
                logging.info(f"阈值优化: 高={old_high:.3f}->{self.fallback_threshold_high:.3f}, "
                           f"中={old_medium:.3f}->{self.fallback_threshold_medium:.3f}, "
                           f"低={old_low:.3f}->{self.fallback_threshold_low:.3f}")
                logging.info(f"基于统计: 平均分数={recent_avg:.3f}, 标准差={recent_std:.3f}, "
                           f"Fallback率={fallback_rate:.1%}")
                
        except Exception as e:
            logging.warning(f"阈值优化失败: {e}")
    
    def _log_score_statistics(self):
        """记录分数统计信息"""
        try:
            stats = self.score_statistics
            
            if stats['avg_scores']:
                recent_avg = np.mean(stats['avg_scores'][-100:])  # 最近100次的平均分数
                overall_avg = np.mean(stats['avg_scores'])
                
                logging.info(f"分数统计 (查询数: {stats['query_count']})")
                logging.info(f"  整体平均分数: {overall_avg:.4f}")
                logging.info(f"  最近平均分数: {recent_avg:.4f}")
                
                if stats['top3_scores']:
                    top3_avg = np.mean(stats['top3_scores'][-100:])
                    logging.info(f"  Top-3平均分数: {top3_avg:.4f}")
                
                fallback_rate = stats['fallback_triggered'] / stats['query_count']
                logging.info(f"  Fallback触发率: {fallback_rate:.1%}")
                logging.info(f"  阈值调整次数: {stats['threshold_adjustments']}")
                
                logging.info(f"当前阈值: 高={self.fallback_threshold_high:.3f}, "
                           f"中={self.fallback_threshold_medium:.3f}, "
                           f"低={self.fallback_threshold_low:.3f}")
                
        except Exception as e:
            logging.warning(f"记录分数统计失败: {e}")
    
    def get_score_statistics(self) -> Dict[str, Any]:
        """获取分数统计信息
        
        Returns:
            分数统计字典
        """
        stats = self.score_statistics.copy()
        
        # 添加计算字段
        if stats['avg_scores']:
            stats['overall_avg_score'] = np.mean(stats['avg_scores'])
            stats['recent_avg_score'] = np.mean(stats['avg_scores'][-100:]) if len(stats['avg_scores']) >= 100 else np.mean(stats['avg_scores'])
        
        if stats['top3_scores']:
             stats['top3_avg_score'] = np.mean(stats['top3_scores'])
        
        if stats['query_count'] > 0:
            stats['fallback_rate'] = stats['fallback_triggered'] / stats['query_count']
        
        # 添加当前阈值
        stats['current_thresholds'] = {
            'high': self.fallback_threshold_high,
            'medium': self.fallback_threshold_medium,
            'low': self.fallback_threshold_low
        }
        
        return stats
    
    def retrieve(
        self, 
        query: str, 
    ) -> List[RAGRetrievalResult]:
        
        if not self.is_index_loaded:
            if not self.load_index():
                logging.error("FAISS索引未加载")
                return []
        
        self.retrieval_count += 1
        
        try:
            # 编码查询（使用BGE查询模式）
            query_embedding = self._encode_texts([query], is_query=True)[0]
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # L2归一化
            faiss.normalize_L2(query_embedding)
            
            # 第一阶段：FAISS检索top-1000候选结果
            # 如果启用reranker，检索更多候选结果用于重排序
            top_candidates = self.top_candidates if self.use_reranker else self.final_top_k
            
            search_start_time = time.time()
            scores, indices = self.index.search(query_embedding, top_candidates)
            search_time = time.time() - search_start_time
            
            # 记录搜索时间
            self.index_performance['search_times'].append(search_time)
            # 保持最近100次搜索记录
            if len(self.index_performance['search_times']) > 100:
                self.index_performance['search_times'] = self.index_performance['search_times'][-100:]
            
            # 构建初步检索结果
            initial_results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score < self.min_score:
                    continue
                
                if idx < len(self.doc_ids):
                    doc_id = self.doc_ids[idx]
                    
                    # 获取chunk metadata
                    chunk_meta = self.chunk_metadata[idx] if hasattr(self, 'chunk_metadata') and idx < len(self.chunk_metadata) else {}
                    
                    # 从数据库获取原始文档
                    original_doc_id = chunk_meta.get('original_doc_id', doc_id)
                    cursor = self.conn.execute(
                        "SELECT title, text, metadata FROM documents WHERE id = ?",
                        (original_doc_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        title, text, metadata = row
                        
                        # 使用chunk文本或原始文本
                        chunk_text = chunk_meta.get('chunk_text', text)
                        
                        # 合并metadata
                        doc_metadata = eval(metadata) if metadata else {}
                        doc_metadata.update(chunk_meta)
                        
                        # 为chunk创建唯一ID
                        chunk_id = chunk_meta.get('chunk_id', 0)
                        unique_chunk_id = f"{original_doc_id}_{chunk_id}"
                        
                        document = RAGDocument(
                            id=unique_chunk_id,
                            title=title,
                            text=chunk_text,
                            metadata=doc_metadata
                        )
                        
                        result = RAGRetrievalResult(
                            document=document,
                            score=float(score),
                            rank=rank
                        )
                        initial_results.append(result)
            
            logging.info(f"第一阶段检索完成，查询: '{query[:]}...'，获得 {len(initial_results)} 个候选结果")
            
            # 第二阶段：BGE reranker重排序（如果启用）
            if self.use_reranker and initial_results:
                logging.info(f"开始重排序: {len(initial_results)} -> {self.final_top_k} 结果")
                final_results = self._rerank_results(query, initial_results)
                # 确保返回的结果数量不超过final_top_k
                final_results = final_results[:self.final_top_k]
            else:
                # 如果未启用reranker，直接返回前final_top_k个结果
                final_results = initial_results[:self.final_top_k]
            
            logging.info(f"检索完成，查询: '{query[:50]}...'，FAISS: {len(initial_results)} -> 最终: {len(final_results)} 个结果")
            
            # 第三阶段：Fallback机制（根据RAG方案）
            if self.enable_fallback and final_results:
                final_results = self._apply_fallback_mechanism(query, final_results, self.final_top_k)
            
            # 第四阶段：分数统计和阈值优化
            self._update_score_statistics(query, final_results)
            
            return final_results
            
        except Exception as e:
            logging.error(f"检索失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": self.total_documents,
            "retrieval_count": self.retrieval_count,
            "index_build_time": self.index_build_time,
            "is_index_loaded": self.is_index_loaded,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（别名方法）"""
        stats = self.get_statistics()
        # 添加测试脚本需要的字段
        stats["index_size"] = len(self.doc_ids) if hasattr(self, 'doc_ids') and self.doc_ids else 0
        return stats
    
    def _apply_fallback_mechanism(self, query: str, local_results: List[RAGRetrievalResult], top_k: int) -> List[RAGRetrievalResult]:
        """应用Fallback机制（根据RAG方案）
        
        根据top-k平均得分判断是否启用网络搜索：
        - ≥0.35: 只用本地结果
        - 0.2-0.35: 启用fallback + 本地混合
        - <0.2: 一定启用fallback
        
        Args:
            query: 查询文本
            local_results: 本地检索结果
            top_k: 最终返回的结果数量
            
        Returns:
            经过Fallback处理的最终结果
        """
        try:
            # 计算top-k平均得分
            top_k_results = local_results[:top_k]
            if not top_k_results:
                return local_results
            
            avg_score = sum(result.score for result in top_k_results) / len(top_k_results)
            logging.info(f"Top-k平均得分: {avg_score:.4f}")
            
            # 根据阈值判断策略
            if avg_score >= self.fallback_threshold_high:
                # 高阈值：只用本地结果
                logging.info("得分较高，使用本地结果")
                return local_results[:top_k]
            
            elif avg_score >= self.fallback_threshold_medium:
                # 中阈值：fallback + 本地混合
                logging.info("得分中等，启用Fallback + 本地混合")
                self._last_fallback_triggered = True  # 标记Fallback被触发
                fallback_results = self._get_fallback_results(query, top_k // 2)
                return self._merge_results(local_results, fallback_results, top_k)
            
            else:
                # 低阈值：一定启用fallback
                logging.info("得分较低，优先使用Fallback结果")
                self._last_fallback_triggered = True  # 标记Fallback被触发
                fallback_results = self._get_fallback_results(query, 6)
                if fallback_results:
                    # 如果有fallback结果，与本地结果混合
                    return self._merge_results(fallback_results, local_results, top_k)
                else:
                    # 如果fallback失败，返回本地结果
                    return local_results[:top_k]
                    
        except Exception as e:
            logging.error(f"Fallback机制执行失败: {e}")
            return local_results[:top_k]
    
    def _get_fallback_results(self, query: str, max_results: int) -> List[RAGRetrievalResult]:
        """获取Fallback结果（网络搜索）
        
        Args:
            query: 查询文本
            max_results: 最大结果数量
            
        Returns:
            网络搜索的结果列表
        """
        try:
            # 首先检查缓存
            cached_results = self._get_cached_fallback_results(query)
            if cached_results:
                logging.info(f"使用缓存的Fallback结果: {len(cached_results)} 个")
                return cached_results[:max_results]
            
            # 执行网络搜索
            web_results = self._search_web_knowledge(query, max_results)
            
            if web_results:
                # 对网络结果进行embedding和reranker处理
                processed_results = self._process_web_results(query, web_results)
                
                # 缓存结果
                self._cache_fallback_results(query, processed_results)
                
                logging.info(f"获取到 {len(processed_results)} 个Fallback结果")
                return processed_results[:max_results]
            
            return []
            
        except Exception as e:
            logging.error(f"获取Fallback结果失败: {e}")
            return []
    
    def _merge_results(self, primary_results: List[RAGRetrievalResult], 
                      secondary_results: List[RAGRetrievalResult], 
                      top_k: int) -> List[RAGRetrievalResult]:
        """合并两组检索结果
        
        Args:
            primary_results: 主要结果
            secondary_results: 次要结果
            top_k: 最终返回的结果数量
            
        Returns:
            合并后的结果列表
        """
        try:
            # 合并所有结果并去重（基于文档ID）
            seen_ids = set()
            all_results = []
            
            # 添加所有主要结果
            for result in primary_results:
                if result.document.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.document.id)
            
            # 添加所有次要结果
            for result in secondary_results:
                if result.document.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.document.id)
            
            # 统一按分数降序排序
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # 选择 top-k 个结果并更新排名
            final_results = all_results[:top_k]
            for i, result in enumerate(final_results):
                result.rank = i
            
            return final_results
            
        except Exception as e:
            logging.error(f"合并结果失败: {e}")
            return primary_results[:top_k]
    
    def _get_cached_fallback_results(self, query: str) -> List[RAGRetrievalResult]:
        """从缓存中获取Fallback结果
        
        Args:
            query: 查询文本
            
        Returns:
            缓存的结果列表
        """
        try:
            import hashlib
            import json
            
            # 生成查询哈希
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            
            # 查询缓存
            current_time = datetime.now().isoformat()
            cursor = self.fallback_conn.execute(
                "SELECT results, score FROM fallback_cache WHERE query_hash = ? AND expires_at > ?",
                (query_hash, current_time)
            )
            row = cursor.fetchone()
            
            if row:
                results_json, score = row
                results_data = json.loads(results_json)
                
                # 重构RAGRetrievalResult对象
                cached_results = []
                for result_data in results_data:
                    doc_data = result_data['document']
                    document = RAGDocument(
                        id=doc_data['id'],
                        title=doc_data['title'],
                        text=doc_data['text'],
                        metadata=doc_data['metadata']
                    )
                    result = RAGRetrievalResult(
                        document=document,
                        score=result_data['score'],
                        rank=result_data['rank']
                    )
                    cached_results.append(result)
                
                return cached_results
            
            return []
            
        except Exception as e:
            logging.warning(f"获取缓存Fallback结果失败: {e}")
            return []
    
    def _cache_fallback_results(self, query: str, results: List[RAGRetrievalResult]):
        """缓存Fallback结果
        
        Args:
            query: 查询文本
            results: 要缓存的结果列表
        """
        try:
            import hashlib
            import json
            
            if not results:
                return
            
            # 生成查询哈希
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            
            # 序列化结果
            results_data = []
            for result in results:
                result_data = {
                    'document': {
                        'id': result.document.id,
                        'title': result.document.title,
                        'text': result.document.text,
                        'metadata': result.document.metadata
                    },
                    'score': result.score,
                    'rank': result.rank
                }
                results_data.append(result_data)
            
            results_json = json.dumps(results_data, ensure_ascii=False)
            
            # 计算平均分数
            avg_score = sum(r.score for r in results) / len(results)
            
            # 设置过期时间
            expires_at = (datetime.now() + timedelta(seconds=self.fallback_cache_ttl)).isoformat()
            
            # 插入或更新缓存
            self.fallback_conn.execute(
                "INSERT OR REPLACE INTO fallback_cache (query_hash, query, results, source, score, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (query_hash, query, results_json, "web_search", avg_score, expires_at)
            )
            self.fallback_conn.commit()
            
            logging.info(f"已缓存 {len(results)} 个Fallback结果")
            
        except Exception as e:
            logging.warning(f"缓存Fallback结果失败: {e}")
    
    def _search_web_knowledge(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """执行网络知识搜索
        
        Args:
            query: 查询文本
            max_results: 最大结果数量
            
        Returns:
            网络搜索结果列表
        """
        try:
            # 这里可以集成多种网络搜索API
            # 1. Wikipedia API
            # 2. Bing Search API
            # 3. Google Custom Search API
            
            web_results = []
            
            # 尝试Wikipedia API搜索
            try:
                wiki_results = self._search_wikipedia_api(query, max_results // 2)
                web_results.extend(wiki_results)
                logging.info(f"Wikipedia API返回 {len(wiki_results)} 个结果")
            except Exception as e:
                logging.warning(f"Wikipedia API搜索失败: {e}")
            
            # 如果结果不足，可以尝试其他搜索引擎
            if len(web_results) < max_results:
                # 这里可以添加其他搜索引擎的调用
                # 例如：bing_results = self._search_bing_api(query, max_results - len(web_results))
                pass
            
            return web_results[:max_results]
            
        except Exception as e:
            logging.error(f"网络知识搜索失败: {e}")
            return []
    
    def _search_wikipedia_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """使用Wikipedia API搜索
        
        Args:
            query: 查询文本
            max_results: 最大结果数量
            
        Returns:
            Wikipedia搜索结果列表
        """
        try:
            if not REQUESTS_AVAILABLE:
                logging.warning("Requests库不可用，无法执行Wikipedia API搜索")
                return []
            
            # Wikipedia API搜索
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            
            # 首先搜索相关页面
            search_api_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results
            }
            
            response = requests.get(search_api_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            results = []
            if 'query' in search_data and 'search' in search_data['query']:
                for item in search_data['query']['search'][:max_results]:
                    title = item['title']
                    snippet = item.get('snippet', '')
                    
                    # 获取页面摘要
                    try:
                        summary_response = requests.get(f"{search_url}{title}", timeout=5)
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            text = summary_data.get('extract', snippet)
                        else:
                            text = snippet
                    except:
                        text = snippet
                    
                    # 清理HTML标签
                    import re
                    text = re.sub(r'<[^>]+>', '', text)
                    
                    result = {
                        'id': f"wiki_{title.replace(' ', '_')}",
                        'title': title,
                        'text': text,
                        'source': 'wikipedia',
                        'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        'metadata': {
                            'source': 'wikipedia_api',
                            'search_query': query
                        }
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Wikipedia API搜索失败: {e}")
            return []
    
    def _process_web_results(self, query: str, web_results: List[Dict[str, Any]]) -> List[RAGRetrievalResult]:
        """处理网络搜索结果
        
        对网络搜索结果进行embedding和reranker处理
        
        Args:
            query: 查询文本
            web_results: 网络搜索结果
            
        Returns:
            处理后的RAGRetrievalResult列表
        """
        try:
            if not web_results:
                return []
            
            # 构建RAGRetrievalResult对象
            rag_results = []
            for i, web_result in enumerate(web_results):
                document = RAGDocument(
                    id=web_result['id'],
                    title=web_result['title'],
                    text=web_result['text'],
                    metadata=web_result.get('metadata', {})
                )
                
                # 初始分数（基于搜索排名）
                initial_score = 1.0 - (i * 0.1)  # 递减分数
                
                result = RAGRetrievalResult(
                    document=document,
                    score=initial_score,
                    rank=i
                )
                rag_results.append(result)
            
            # 如果启用了reranker，对网络结果进行重排序
            if self.use_reranker and rag_results:
                logging.info(f"对 {len(rag_results)} 个网络结果进行重排序")
                rag_results = self._rerank_results(query, rag_results)
            
            return rag_results
            
        except Exception as e:
            logging.error(f"处理网络搜索结果失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'fallback_conn'):
            self.fallback_conn.close()