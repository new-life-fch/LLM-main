"""
CausalEditor主类
层级化因果溯源与反事实编辑的核心实现
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .conflict_detector import CausalConflictDetector
from .counterfactual_editor import CounterfactualEditor
from .vector_database import VectorDatabase


class CausalEditor:
    """
    CausalEditor: 层级化因果溯源与反事实编辑
    
    核心思想：通过预计算的因果知识图谱激活编码，在推理时检测因果冲突
    并进行外科手术式的反事实编辑，以修正LLM的幻觉问题。
    """
    
    def __init__(
        self,
        vector_db_path: str,
        model_name: str = "llama-2-7b",
        edit_strength: float = 1.0,
        top_layers: int = 10,
        similarity_threshold: float = 0.8,
        conflict_threshold: float = 0.6,
        device: str = "cuda"
    ):
        """
        初始化CausalEditor
        
        Args:
            vector_db_path: 预计算的向量数据库路径
            model_name: 基座模型名称
            edit_strength: 编辑强度
            top_layers: 参与编辑的顶层数量
            similarity_threshold: 相似度阈值
            conflict_threshold: 冲突检测阈值
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.edit_strength = edit_strength
        self.top_layers = top_layers
        
        # 初始化组件
        self.vector_db = VectorDatabase(vector_db_path, device=self.device)
        self.conflict_detector = CausalConflictDetector(
            vector_db=self.vector_db,
            similarity_threshold=similarity_threshold,
            conflict_threshold=conflict_threshold
        )
        self.counterfactual_editor = CounterfactualEditor(
            edit_strength=edit_strength,
            device=self.device
        )
        
        # 运行时状态
        self.current_layer_id = None
        self.prompt_length = None
        self.is_mc_mode = False  # multiple choice模式
        
        # 层级映射（不同模型可能有不同的层级结构）
        self.layer_mapping = self._get_layer_mapping(model_name)
        
        logging.info(f"CausalEditor初始化完成: {model_name}, 编辑强度: {edit_strength}")
    
    def _get_layer_mapping(self, model_name: str) -> Dict[str, int]:
        """
        获取模型层级映射
        不同模型的层级结构可能不同，需要适配
        """
        if "llama" in model_name.lower():
            return {"attn_factor": 2, "mlp_offset": 1}
        elif "mistral" in model_name.lower():
            return {"attn_factor": 2, "mlp_offset": 1}
        else:
            # 默认映射
            return {"attn_factor": 2, "mlp_offset": 1}
    
    def set_generation_mode(self, is_mc: bool = False, prompt_length: Optional[int] = None):
        """
        设置生成模式
        
        Args:
            is_mc: 是否为multiple choice模式
            prompt_length: 提示长度（MC模式需要）
        """
        self.is_mc_mode = is_mc
        self.prompt_length = prompt_length
        logging.info(f"设置生成模式: MC={is_mc}, prompt_length={prompt_length}")
    
    def set_current_layer(self, layer_id: str):
        """
        设置当前处理的层
        
        Args:
            layer_id: 层ID，格式如 "10.attn" 或 "10.mlp"
        """
        self.current_layer_id = layer_id
    
    def _get_layer_rank(self, layer_id: str) -> int:
        """
        将层ID转换为层级排名
        
        Args:
            layer_id: 层ID
            
        Returns:
            层级排名数字
        """
        try:
            layer_num = int(layer_id.split(".")[0])
            if layer_id.endswith("attn"):
                return self.layer_mapping["attn_factor"] * layer_num
            else:  # mlp
                return self.layer_mapping["attn_factor"] * layer_num + self.layer_mapping["mlp_offset"]
        except (ValueError, IndexError):
            logging.warning(f"无法解析层ID: {layer_id}")
            return 0
    
    def should_edit_layer(self, layer_id: str) -> bool:
        """
        判断是否应该编辑当前层
        
        Args:
            layer_id: 层ID
            
        Returns:
            是否应该编辑
        """
        layer_rank = self._get_layer_rank(layer_id)
        return layer_rank <= self.top_layers
    
    @torch.inference_mode()
    def edit_activations(
        self, 
        activations: torch.Tensor,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        编辑激活状态的主入口函数
        
        Args:
            activations: 当前激活状态 [batch_size, seq_len, hidden_dim]
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            
        Returns:
            编辑后的激活状态
        """
        if self.current_layer_id is None:
            logging.warning("当前层ID未设置，跳过编辑")
            return activations
        
        # 检查是否应该编辑当前层
        if not self.should_edit_layer(self.current_layer_id):
            return activations
        
        batch_size, seq_len, hidden_dim = activations.shape
        
        # 检测因果冲突
        conflict_info = self.conflict_detector.detect_conflict(
            activations=activations,
            generated_tokens=generated_tokens,
            context_tokens=context_tokens,
            layer_id=self.current_layer_id
        )
        
        if not conflict_info["has_conflict"]:
            # 没有检测到冲突，直接返回原激活
            return activations
        
        # 执行反事实编辑
        edited_activations = self.counterfactual_editor.edit(
            activations=activations,
            conflict_info=conflict_info,
            is_mc_mode=self.is_mc_mode,
            prompt_length=self.prompt_length
        )
        
        logging.debug(f"层 {self.current_layer_id} 检测到冲突并完成编辑")
        return edited_activations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取运行统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "model_name": self.model_name,
            "edit_strength": self.edit_strength,
            "top_layers": self.top_layers,
            "vector_db_size": self.vector_db.get_size(),
            "conflict_detector_stats": self.conflict_detector.get_statistics(),
            "counterfactual_editor_stats": self.counterfactual_editor.get_statistics()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.conflict_detector.reset_statistics()
        self.counterfactual_editor.reset_statistics()
    
    def save_config(self, path: str):
        """
        保存配置
        
        Args:
            path: 保存路径
        """
        config = {
            "model_name": self.model_name,
            "edit_strength": self.edit_strength,
            "top_layers": self.top_layers,
            "similarity_threshold": self.conflict_detector.similarity_threshold,
            "conflict_threshold": self.conflict_detector.conflict_threshold,
            "layer_mapping": self.layer_mapping
        }
        
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logging.info(f"配置已保存到: {path}")
    
    @classmethod
    def from_config(cls, config_path: str, vector_db_path: str, device: str = "cuda"):
        """
        从配置文件加载CausalEditor
        
        Args:
            config_path: 配置文件路径
            vector_db_path: 向量数据库路径
            device: 计算设备
            
        Returns:
            CausalEditor实例
        """
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(
            vector_db_path=vector_db_path,
            model_name=config["model_name"],
            edit_strength=config["edit_strength"],
            top_layers=config["top_layers"],
            similarity_threshold=config["similarity_threshold"],
            conflict_threshold=config["conflict_threshold"],
            device=device
        ) 