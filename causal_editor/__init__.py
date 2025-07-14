"""
CausalEditor: 层级化因果溯源与反事实编辑
一种通过预计算因果知识图谱激活编码，在推理时进行精确反事实编辑的方法
"""

__version__ = "0.1.0"
__author__ = "CausalEditor Team"

from .core.causal_editor import CausalEditor
from .core.conflict_detector import CausalConflictDetector  
from .core.counterfactual_editor import CounterfactualEditor
from .core.vector_database import VectorDatabase

__all__ = [
    "CausalEditor",
    "CausalConflictDetector", 
    "CounterfactualEditor",
    "VectorDatabase"
] 