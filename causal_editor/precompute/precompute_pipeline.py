"""
预计算流水线
整合知识提取、激活指纹构建和向量数据库建立的完整流程
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from .knowledge_extractor import KnowledgeExtractor, WikidataExtractor, CSVKnowledgeExtractor
from .fingerprint_builder import ActivationFingerprintBuilder
from ..core.vector_database import VectorDatabase


class PrecomputePipeline:
    """
    预计算流水线
    执行完整的知识提取到激活指纹构建的流程
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str = "./precomputed_data",
        device: str = "cuda",
        batch_size: int = 8
    ):
        """
        初始化预计算流水线
        
        Args:
            model_name: 目标LLM模型名称
            output_dir: 输出目录
            device: 计算设备
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        
        # 初始化组件
        self.fingerprint_builder = None
        self.vector_db = None
        
        # 配置日志
        self.log_file = self.output_dir / "precompute.log"
        self._setup_logging()
        
        logging.info(f"预计算流水线初始化完成，输出目录: {output_dir}")
    
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_full_pipeline(
        self,
        knowledge_config: Dict[str, Any],
        limit: int = 10000,
        target_layers: Optional[List[int]] = None
    ) -> str:
        """
        运行完整的预计算流水线
        
        Args:
            knowledge_config: 知识提取配置
            limit: 三元组数量限制
            target_layers: 目标层列表
            
        Returns:
            向量数据库路径
        """
        start_time = time.time()
        
        try:
            # 步骤1: 提取知识三元组
            logging.info("=" * 50)
            logging.info("步骤1: 提取知识三元组")
            knowledge_triplets = self._extract_knowledge(knowledge_config, limit)
            
            # 步骤2: 构建激活指纹
            logging.info("=" * 50)
            logging.info("步骤2: 构建激活指纹")
            fingerprints = self._build_fingerprints(knowledge_triplets, target_layers)
            
            # 步骤3: 构建向量数据库
            logging.info("=" * 50)
            logging.info("步骤3: 构建向量数据库")
            db_path = self._build_vector_database(fingerprints, knowledge_triplets)
            
            # 步骤4: 验证和统计
            logging.info("=" * 50)
            logging.info("步骤4: 验证和统计")
            self._generate_statistics(db_path, knowledge_triplets)
            
            total_time = time.time() - start_time
            logging.info(f"预计算流水线完成，总耗时: {total_time:.2f}秒")
            
            return str(db_path)
            
        except Exception as e:
            logging.error(f"预计算流水线失败: {e}")
            raise
    
    def _extract_knowledge(
        self,
        config: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        提取知识三元组
        
        Args:
            config: 提取配置
            limit: 数量限制
            
        Returns:
            知识三元组列表
        """
        # 创建知识提取器
        extractor_type = config.get('type', 'wikidata')
        
        if extractor_type == 'wikidata':
            extractor = WikidataExtractor(
                cache_dir=str(self.output_dir / "wikidata_cache"),
                rate_limit=config.get('rate_limit', 1.0)
            )
        elif extractor_type == 'csv':
            extractor = CSVKnowledgeExtractor(config['csv_path'])
        else:
            raise ValueError(f"不支持的提取器类型: {extractor_type}")
        
        # 提取三元组
        knowledge_triplets = extractor.extract_triplets(limit=limit)
        
        # 保存三元组
        triplets_file = self.output_dir / "knowledge_triplets.json"
        with open(triplets_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_triplets, f, ensure_ascii=False, indent=2)
        
        logging.info(f"提取了 {len(knowledge_triplets)} 个知识三元组")
        logging.info(f"三元组已保存到: {triplets_file}")
        
        return knowledge_triplets
    
    def _build_fingerprints(
        self,
        knowledge_triplets: List[Dict[str, Any]],
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        构建激活指纹
        
        Args:
            knowledge_triplets: 知识三元组
            target_layers: 目标层
            
        Returns:
            层级激活指纹
        """
        # 初始化指纹构建器
        if self.fingerprint_builder is None:
            self.fingerprint_builder = ActivationFingerprintBuilder(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                target_layers=target_layers,
                cache_dir=str(self.output_dir / "fingerprint_cache")
            )
        
        # 构建指纹
        fingerprints_file = self.output_dir / "activation_fingerprints.npz"
        fingerprints = self.fingerprint_builder.build_fingerprints(
            knowledge_triplets=knowledge_triplets,
            output_path=str(fingerprints_file)
        )
        
        logging.info(f"激活指纹已保存到: {fingerprints_file}")
        
        # 保存层级维度信息
        dimensions = self.fingerprint_builder.get_layer_dimensions()
        dimensions_file = self.output_dir / "layer_dimensions.json"
        with open(dimensions_file, 'w', encoding='utf-8') as f:
            json.dump(dimensions, f, indent=2)
        
        return fingerprints
    
    def _build_vector_database(
        self,
        fingerprints: Dict[str, np.ndarray],
        knowledge_triplets: List[Dict[str, Any]]
    ) -> Path:
        """
        构建向量数据库
        
        Args:
            fingerprints: 激活指纹
            knowledge_triplets: 知识三元组
            
        Returns:
            数据库路径
        """
        db_path = self.output_dir / "vector_database"
        
        # 初始化向量数据库
        self.vector_db = VectorDatabase(str(db_path), device=self.device)
        
        # 分层添加向量
        for layer_id, layer_fingerprints in fingerprints.items():
            if len(layer_fingerprints) == 0:
                logging.warning(f"层 {layer_id} 没有激活指纹，跳过")
                continue
            
            logging.info(f"添加层 {layer_id} 的 {len(layer_fingerprints)} 个向量到数据库")
            
            self.vector_db.add_vectors(
                vectors=layer_fingerprints,
                layer_id=layer_id,
                knowledge_triplets=knowledge_triplets
            )
        
        # 保存数据库
        self.vector_db.save_database()
        
        # 优化索引
        self.vector_db.optimize_index()
        
        logging.info(f"向量数据库已保存到: {db_path}")
        return db_path
    
    def _generate_statistics(
        self,
        db_path: Path,
        knowledge_triplets: List[Dict[str, Any]]
    ):
        """
        生成统计报告
        
        Args:
            db_path: 数据库路径
            knowledge_triplets: 知识三元组
        """
        # 数据库统计
        db_stats = self.vector_db.get_statistics() if self.vector_db else {}
        
        # 知识统计
        knowledge_stats = self._analyze_knowledge_distribution(knowledge_triplets)
        
        # 模型统计
        model_stats = {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size
        }
        
        # 综合统计
        statistics = {
            'model_info': model_stats,
            'knowledge_stats': knowledge_stats,
            'database_stats': db_stats,
            'precompute_timestamp': time.time()
        }
        
        # 保存统计报告
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        self._generate_readable_report(statistics)
        
        logging.info(f"统计报告已保存到: {stats_file}")
    
    def _analyze_knowledge_distribution(
        self,
        knowledge_triplets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析知识分布
        
        Args:
            knowledge_triplets: 知识三元组
            
        Returns:
            知识分布统计
        """
        if not knowledge_triplets:
            return {}
        
        # 关系分布
        relations = [t.get('relation', '') for t in knowledge_triplets]
        relation_counts = {}
        for relation in relations:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        # 来源分布
        sources = [t.get('source', 'unknown') for t in knowledge_triplets]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # 置信度分布
        confidences = [t.get('confidence', 1.0) for t in knowledge_triplets]
        
        return {
            'total_triplets': len(knowledge_triplets),
            'unique_relations': len(relation_counts),
            'relation_distribution': dict(sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'source_distribution': source_counts,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
    
    def _generate_readable_report(self, statistics: Dict[str, Any]):
        """
        生成可读的统计报告
        
        Args:
            statistics: 统计数据
        """
        report_file = self.output_dir / "precompute_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CausalEditor 预计算报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型信息
            model_info = statistics.get('model_info', {})
            f.write("模型信息:\n")
            f.write(f"  模型名称: {model_info.get('model_name', 'N/A')}\n")
            f.write(f"  计算设备: {model_info.get('device', 'N/A')}\n")
            f.write(f"  批处理大小: {model_info.get('batch_size', 'N/A')}\n\n")
            
            # 知识统计
            knowledge_stats = statistics.get('knowledge_stats', {})
            f.write("知识统计:\n")
            f.write(f"  总三元组数: {knowledge_stats.get('total_triplets', 0)}\n")
            f.write(f"  唯一关系数: {knowledge_stats.get('unique_relations', 0)}\n")
            
            # 关系分布
            relation_dist = knowledge_stats.get('relation_distribution', {})
            if relation_dist:
                f.write("  热门关系:\n")
                for relation, count in list(relation_dist.items())[:10]:
                    f.write(f"    {relation}: {count}\n")
            
            # 数据库统计
            f.write("\n向量数据库统计:\n")
            db_stats = statistics.get('database_stats', {})
            f.write(f"  总向量数: {db_stats.get('total_vectors', 0)}\n")
            f.write(f"  向量维度: {db_stats.get('dimension', 0)}\n")
            
            layer_stats = db_stats.get('layer_statistics', {})
            if layer_stats:
                f.write("  各层向量数:\n")
                for layer, count in layer_stats.items():
                    f.write(f"    {layer}: {count}\n")
        
        logging.info(f"可读报告已保存到: {report_file}")
    
    def run_incremental_update(
        self,
        new_knowledge_config: Dict[str, Any],
        existing_db_path: str,
        limit: int = 1000
    ) -> str:
        """
        增量更新已存在的数据库
        
        Args:
            new_knowledge_config: 新知识提取配置
            existing_db_path: 现有数据库路径
            limit: 新增三元组限制
            
        Returns:
            更新后的数据库路径
        """
        logging.info("开始增量更新")
        
        # 加载现有数据库
        self.vector_db = VectorDatabase(existing_db_path, device=self.device)
        
        # 提取新知识
        new_triplets = self._extract_knowledge(new_knowledge_config, limit)
        
        # 构建新指纹
        if self.fingerprint_builder is None:
            self.fingerprint_builder = ActivationFingerprintBuilder(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                cache_dir=str(self.output_dir / "fingerprint_cache")
            )
        
        new_fingerprints = self.fingerprint_builder.build_fingerprints(new_triplets)
        
        # 更新数据库
        for layer_id, layer_fingerprints in new_fingerprints.items():
            if len(layer_fingerprints) > 0:
                self.vector_db.add_vectors(
                    vectors=layer_fingerprints,
                    layer_id=layer_id,
                    knowledge_triplets=new_triplets
                )
        
        # 保存更新后的数据库
        self.vector_db.save_database()
        self.vector_db.optimize_index()
        
        logging.info("增量更新完成")
        return existing_db_path
    
    def validate_database(self, db_path: str) -> Dict[str, Any]:
        """
        验证数据库完整性
        
        Args:
            db_path: 数据库路径
            
        Returns:
            验证结果
        """
        try:
            # 加载数据库
            test_db = VectorDatabase(db_path, device=self.device)
            
            # 基本统计
            stats = test_db.get_statistics()
            
            # 搜索测试
            if stats['total_vectors'] > 0:
                # 创建一个随机查询向量
                dimension = stats['dimension']
                if dimension and dimension > 0:
                    test_vector = np.random.randn(dimension).astype(np.float32)
                    
                    # 执行搜索测试
                    import torch
                    test_results = test_db.search(
                        query_vector=torch.from_numpy(test_vector),
                        k=5
                    )
                    
                    validation_result = {
                        'status': 'success',
                        'total_vectors': stats['total_vectors'],
                        'dimension': stats['dimension'],
                        'search_test_results': len(test_results),
                        'layer_counts': stats['layer_statistics']
                    }
                else:
                    validation_result = {
                        'status': 'error',
                        'message': 'Invalid dimension'
                    }
            else:
                validation_result = {
                    'status': 'warning',
                    'message': 'No vectors in database'
                }
            
            return validation_result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def cleanup_cache(self):
        """清理缓存文件"""
        cache_dirs = [
            self.output_dir / "wikidata_cache",
            self.output_dir / "fingerprint_cache"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logging.info(f"清理缓存目录: {cache_dir}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        获取配置模板
        
        Returns:
            配置模板
        """
        return {
            "knowledge_extraction": {
                "type": "wikidata",  # or "csv"
                "rate_limit": 1.0,
                # "csv_path": "path/to/knowledge.csv"  # for CSV type
            },
            "model": {
                "name": "meta-llama/Llama-2-7b-hf",
                "device": "cuda",
                "batch_size": 8
            },
            "fingerprint": {
                "target_layers": None,  # None for all layers, or [10, 11, 12, ...]
                "cache_activations": True
            },
            "database": {
                "index_type": "HNSW",
                "similarity_metric": "cosine"
            },
            "limits": {
                "max_triplets": 10000,
                "max_layers": 32
            }
        } 