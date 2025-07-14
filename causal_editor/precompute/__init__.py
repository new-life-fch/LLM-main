"""
预计算模块
负责知识提取、激活指纹构建和向量数据库建立
"""

from .knowledge_extractor import WikidataExtractor, KnowledgeExtractor
from .fingerprint_builder import ActivationFingerprintBuilder
from .precompute_pipeline import PrecomputePipeline

__all__ = [
    "WikidataExtractor",
    "KnowledgeExtractor", 
    "ActivationFingerprintBuilder",
    "PrecomputePipeline"
] 