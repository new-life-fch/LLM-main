"""
反事实激活编辑组件
执行精确的"外科手术式"激活状态编辑
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class CounterfactualEditor:
    """
    反事实激活编辑器
    基于检测到的因果冲突进行精确的激活状态编辑
    """
    
    def __init__(
        self,
        edit_strength: float = 1.0,
        min_confidence: float = 0.5,
        device: str = "cuda"
    ):
        """
        初始化反事实编辑器
        
        Args:
            edit_strength: 编辑强度系数
            min_confidence: 最小置信度阈值
            device: 计算设备
        """
        self.edit_strength = edit_strength
        self.min_confidence = min_confidence
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 统计信息
        self.edit_count = 0
        self.successful_edits = 0
        self.layer_edits = defaultdict(int)
        self.edit_magnitudes = []
        
        logging.info(f"反事实编辑器初始化完成，编辑强度: {edit_strength}")
    
    def edit(
        self,
        activations: torch.Tensor,
        conflict_info: Dict[str, Any],
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        执行反事实编辑
        
        Args:
            activations: 原始激活状态 [batch_size, seq_len, hidden_dim]
            conflict_info: 冲突信息
            is_mc_mode: 是否为multiple choice模式
            prompt_length: 提示长度
            
        Returns:
            编辑后的激活状态
        """
        if not conflict_info.get('has_conflict', False):
            return activations
        
        if conflict_info.get('confidence', 0.0) < self.min_confidence:
            logging.debug("冲突置信度过低，跳过编辑")
            return activations
        
        self.edit_count += 1
        layer_id = conflict_info.get('layer_id', 'unknown')
        self.layer_edits[layer_id] += 1
        
        try:
            # 获取正确的激活目标
            correct_activation = self._get_correct_activation(conflict_info)
            if correct_activation is None:
                return activations
            
            # 执行编辑
            edited_activations = self._apply_counterfactual_edit(
                activations=activations,
                correct_activation=correct_activation,
                conflict_info=conflict_info,
                is_mc_mode=is_mc_mode,
                prompt_length=prompt_length
            )
            
            self.successful_edits += 1
            logging.debug(f"成功编辑层 {layer_id}，置信度: {conflict_info.get('confidence', 0.0):.3f}")
            
            return edited_activations
            
        except Exception as e:
            logging.error(f"编辑失败: {e}")
            return activations
    
    def _get_correct_activation(self, conflict_info: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        获取正确的激活目标
        
        Args:
            conflict_info: 冲突信息
            
        Returns:
            正确的激活向量
        """
        retrieved_knowledge = conflict_info.get('retrieved_knowledge', [])
        correct_object = conflict_info.get('correct_object')
        
        if not retrieved_knowledge or not correct_object:
            return None
        
        # 查找与正确答案匹配的知识激活
        matching_activations = []
        
        for knowledge in retrieved_knowledge:
            if knowledge.get('object', '').lower().strip() == correct_object.lower().strip():
                # 这里需要从vector_db中获取对应的激活向量
                # 由于FAISS不直接存储向量，我们使用检索到的相似度作为权重
                similarity = knowledge.get('similarity_score', 0.0)
                confidence = knowledge.get('confidence', 1.0)
                weight = similarity * confidence
                matching_activations.append((knowledge, weight))
        
        if not matching_activations:
            # 如果没有完全匹配，使用最相似的高置信度激活
            best_knowledge = max(
                retrieved_knowledge,
                key=lambda x: x.get('similarity_score', 0.0) * x.get('confidence', 1.0)
            )
            return self._construct_target_activation(best_knowledge)
        
        # 使用加权平均的方式构建目标激活
        return self._construct_weighted_target_activation(matching_activations)
    
    def _construct_target_activation(self, knowledge: Dict[str, Any]) -> torch.Tensor:
        """
        基于知识信息构建目标激活
        
        由于我们没有直接存储原始激活向量，这里使用启发式方法
        在实际实现中，应该在预计算阶段存储激活向量
        
        Args:
            knowledge: 知识信息
            
        Returns:
            目标激活向量
        """
        # 这里是一个简化的实现
        # 实际应该从预存储的激活向量中获取
        
        # 使用随机向量作为占位符，实际实现时需要替换
        hidden_dim = 4096  # 假设的隐藏维度，应该从实际模型获取
        
        # 基于similarity_score和confidence构建一个方向向量
        similarity = knowledge.get('similarity_score', 0.0)
        confidence = knowledge.get('confidence', 1.0)
        
        # 生成一个基于知识内容的确定性向量
        text_content = f"{knowledge.get('subject', '')} {knowledge.get('relation', '')} {knowledge.get('object', '')}"
        
        # 使用文本内容的hash作为种子，生成确定性的向量
        import hashlib
        seed = int(hashlib.md5(text_content.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        
        target_vector = torch.randn(hidden_dim, device=self.device)
        target_vector = F.normalize(target_vector, p=2, dim=0)
        
        # 根据置信度调整向量强度
        target_vector = target_vector * confidence * similarity
        
        return target_vector
    
    def _construct_weighted_target_activation(
        self, 
        matching_activations: List[Tuple[Dict[str, Any], float]]
    ) -> torch.Tensor:
        """
        构建加权目标激活
        
        Args:
            matching_activations: 匹配的激活信息列表
            
        Returns:
            加权目标激活向量
        """
        if not matching_activations:
            return None
        
        # 计算权重总和
        total_weight = sum(weight for _, weight in matching_activations)
        if total_weight == 0:
            return None
        
        # 构建加权平均向量
        weighted_sum = None
        
        for knowledge, weight in matching_activations:
            target_vector = self._construct_target_activation(knowledge)
            
            if weighted_sum is None:
                weighted_sum = target_vector * (weight / total_weight)
            else:
                weighted_sum += target_vector * (weight / total_weight)
        
        return weighted_sum
    
    def _apply_counterfactual_edit(
        self,
        activations: torch.Tensor,
        correct_activation: torch.Tensor,
        conflict_info: Dict[str, Any],
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        应用反事实编辑
        
        Args:
            activations: 原始激活 [batch_size, seq_len, hidden_dim]
            correct_activation: 正确的激活向量 [hidden_dim]
            conflict_info: 冲突信息
            is_mc_mode: 是否为MC模式
            prompt_length: 提示长度
            
        Returns:
            编辑后的激活
        """
        batch_size, seq_len, hidden_dim = activations.shape
        
        # 获取错误激活（当前生成位置的激活）
        error_activation = activations[:, -1, :]  # [batch_size, hidden_dim]
        
        # 计算编辑向量
        if correct_activation.dim() == 1:
            correct_activation = correct_activation.unsqueeze(0)  # [1, hidden_dim]
        
        # 计算误差投影
        error_projection = self._compute_error_projection(
            error_activation, correct_activation
        )
        
        # 计算编辑delta
        delta = correct_activation - error_projection
        delta = F.normalize(delta, p=2, dim=-1) * torch.norm(error_activation, p=2, dim=-1, keepdim=True)
        
        # 根据置信度调整编辑强度
        confidence = conflict_info.get('confidence', 1.0)
        edit_strength = self.edit_strength * confidence
        
        # 创建编辑掩码
        edit_mask = self._create_edit_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            is_mc_mode=is_mc_mode,
            prompt_length=prompt_length,
            conflict_position=conflict_info.get('conflict_position')
        )
        
        # 应用编辑
        edited_activations = activations.clone()
        delta_expanded = delta.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        edited_activations += delta_expanded * edit_strength * edit_mask.unsqueeze(-1)
        
        # 记录编辑幅度
        edit_magnitude = torch.norm(delta * edit_strength).item()
        self.edit_magnitudes.append(edit_magnitude)
        
        return edited_activations
    
    def _compute_error_projection(
        self, 
        error_activation: torch.Tensor, 
        correct_activation: torch.Tensor
    ) -> torch.Tensor:
        """
        计算错误激活在正确方向上的投影
        
        Args:
            error_activation: 错误激活 [batch_size, hidden_dim]
            correct_activation: 正确激活 [1, hidden_dim]
            
        Returns:
            投影结果
        """
        # 计算投影
        correct_norm = F.normalize(correct_activation, p=2, dim=-1)
        projection_coeff = torch.sum(error_activation * correct_norm, dim=-1, keepdim=True)
        projection = projection_coeff * correct_norm
        
        return projection
    
    def _create_edit_mask(
        self,
        batch_size: int,
        seq_len: int,
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None,
        conflict_position: Optional[int] = None
    ) -> torch.Tensor:
        """
        创建编辑掩码
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            is_mc_mode: 是否为MC模式
            prompt_length: 提示长度
            conflict_position: 冲突位置
            
        Returns:
            编辑掩码 [batch_size, seq_len]
        """
        mask = torch.zeros((batch_size, seq_len), device=self.device)
        
        if is_mc_mode and prompt_length is not None:
            # MC模式：只编辑答案部分
            mask[:, prompt_length + 1:] = 1.0
        else:
            # 生成模式：只编辑最后一个token
            mask[:, -1:] = 1.0
        
        return mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_magnitude = np.mean(self.edit_magnitudes) if self.edit_magnitudes else 0.0
        success_rate = self.successful_edits / max(self.edit_count, 1)
        
        return {
            'edit_count': self.edit_count,
            'successful_edits': self.successful_edits,
            'success_rate': success_rate,
            'average_edit_magnitude': avg_magnitude,
            'layer_edits': dict(self.layer_edits),
            'edit_strength': self.edit_strength,
            'min_confidence': self.min_confidence
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.edit_count = 0
        self.successful_edits = 0
        self.layer_edits.clear()
        self.edit_magnitudes.clear()
    
    def update_edit_strength(self, new_strength: float):
        """
        更新编辑强度
        
        Args:
            new_strength: 新的编辑强度
        """
        self.edit_strength = new_strength
        logging.info(f"编辑强度更新为: {new_strength}")
    
    def get_edit_magnitude_stats(self) -> Dict[str, float]:
        """
        获取编辑幅度统计
        
        Returns:
            编辑幅度统计信息
        """
        if not self.edit_magnitudes:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        magnitudes = np.array(self.edit_magnitudes)
        return {
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes)),
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'count': len(magnitudes)
        } 