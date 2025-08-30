"""
CausalEditor: 动态因果溯源与反事实编辑
一种通过动态指纹生成和实时冲突检测，在推理时进行精确反事实编辑的方法
"""

__version__ = "0.1.0"
__author__ = "CausalEditor Team"

from .core.causal_editor import CausalEditor
from .core.conflict_detector import CausalConflictDetector  
from .core.counterfactual_editor import CounterfactualEditor


__all__ = [
    "CausalEditor",
    "CausalConflictDetector", 
    "CounterfactualEditor"
]