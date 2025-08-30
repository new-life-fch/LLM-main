"""
反事实激活编辑组件
执行精确的"外科手术式"激活状态编辑
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np



class CounterfactualEditor:
    """
    反事实激活编辑器
    基于检测到的因果冲突进行精确的激活状态编辑
    """

    def __init__(
        self,
        edit_strength: float = 1.0,
        min_confidence: float = 0.3,  # 降低最小置信度阈值 # TODO: 需要调参
        device: str = "cuda",
        hidden_size: int = 4096,
        # RAG模式新增参数
        enable_rag_editing: bool = True,
        activation_rollback_strength: float = 0.8,
        resampling_temperature: float = 1.2,
        activation_weighting_factor: float = 0.6,
    ):
        """
        初始化反事实编辑器

        Args:
            edit_strength: 编辑强度系数
            min_confidence: 最小置信度阈值
            device: 计算设备
            hidden_size: 模型隐藏维度
            enable_rag_editing: 是否启用RAG编辑模式
            activation_rollback_strength: 激活回退强度
            resampling_temperature: 重新采样温度
            activation_weighting_factor: 激活加权因子
        """
        self.edit_strength = edit_strength
        self.min_confidence = min_confidence
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        
        # RAG模式参数
        self.enable_rag_editing = enable_rag_editing
        self.activation_rollback_strength = activation_rollback_strength
        self.resampling_temperature = resampling_temperature
        self.activation_weighting_factor = activation_weighting_factor

        # 统计信息
        self.edit_count = 0
        self.successful_edits = 0
        self.layer_edits = defaultdict(int)
        self.edit_magnitudes = []  # 编辑幅度统计
        self.rag_edit_count = 0  # RAG编辑次数
        self.edit_method_stats = defaultdict(int)  # 编辑方法统计

        # 反事实编辑器初始化完成

    def edit(
        self,
        activations: torch.Tensor,
        conflict_info: Dict[str, Any],
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None,
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
        if not conflict_info.get("has_conflict", False):
            return activations

        if conflict_info.get("confidence", 0.0) < self.min_confidence:
            # 冲突置信度过低，跳过编辑
            return activations

        self.edit_count += 1
        layer_id = conflict_info.get("layer_id", "unknown")
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
                prompt_length=prompt_length,
            )

            self.successful_edits += 1
            # 成功编辑层

            return edited_activations

        except Exception as e:
            logging.error(f"编辑失败: {e}")
            return activations

    def _get_correct_activation(
        self, conflict_info: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """
        获取正确的激活目标

        Args:
            conflict_info: 冲突信息

        Returns:
            正确的激活向量
        """
        retrieved_knowledge = conflict_info.get("retrieved_knowledge", [])
        
        # 检查是否有检索到的知识
        if not retrieved_knowledge:
            logging.warning("没有检索到的知识信息，无法构建目标激活")
            return None

        # 如果有correction_vector，优先使用它来构建目标激活
        correction_vector = conflict_info.get("correction_vector")
        if correction_vector is not None:
            # 使用correction_vector作为基础，结合retrieved_knowledge的信息进行调整
            best_knowledge = max(
                retrieved_knowledge,
                key=lambda x: x.get("similarity_score", 0.0) * x.get("confidence", 1.0),
            ) # TODO:可调整的权重
            
            # 获取最佳知识的权重
            similarity = best_knowledge.get("similarity_score", 0.0)
            confidence = best_knowledge.get("confidence", 1.0)
            weight = similarity * confidence
            
            # 将correction_vector作为目标激活的基础，并根据权重进行调整
            target_activation = correction_vector * weight
            
            logging.debug(f"使用correction_vector构建目标激活，权重: {weight:.3f}")
            return target_activation
        

    def _apply_counterfactual_edit(
        self,
        activations: torch.Tensor,
        correct_activation: torch.Tensor,
        conflict_info: Dict[str, Any],
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None,
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
        delta = F.normalize(delta, p=2, dim=-1) * torch.norm(
            error_activation, p=2, dim=-1, keepdim=True
        )

        # 根据置信度调整编辑强度
        confidence = conflict_info.get("confidence", 1.0)
        edit_strength = self.edit_strength * confidence

        # 创建编辑掩码
        edit_mask = self._create_edit_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            is_mc_mode=is_mc_mode,
            prompt_length=prompt_length,
            conflict_position=conflict_info.get("conflict_position"),
        )

        # 应用编辑
        edited_activations = activations.clone()
        delta_expanded = delta.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch_size, seq_len, hidden_dim]

        edited_activations += delta_expanded * edit_strength * edit_mask.unsqueeze(-1)

        # 记录编辑幅度
        edit_magnitude = torch.norm(delta * edit_strength).item()
        self.edit_magnitudes.append(edit_magnitude)

        return edited_activations
    
    def edit_rag(
        self,
        activations: torch.Tensor,
        conflict_info: Dict[str, Any],
        fragment_fingerprints: Dict[str, Dict[str, torch.Tensor]],
        fragment_scores: Dict[str, float],
        layer_id: str = "unknown",
    ) -> torch.Tensor:
        """RAG模式的激活编辑
        
        Args:
            activations: 原始激活状态 [batch_size, seq_len, hidden_dim]
            conflict_info: 冲突信息
            fragment_fingerprints: 片段指纹字典 {fragment_id: {layer_id: tensor}}
            fragment_scores: 片段检索分数字典 {fragment_id: score}
            layer_id: 当前层ID
            
        Returns:
            编辑后的激活状态
        """
        if not self.enable_rag_editing:
            return self.edit(activations, conflict_info)
        
        if not conflict_info.get("has_conflict", False):
            return activations
            
        if conflict_info.get("confidence", 0.0) < self.min_confidence:
            return activations
        
        self.edit_count += 1
        self.rag_edit_count += 1
        self.layer_edits[layer_id] += 1
        
        try:
            # 根据冲突类型选择编辑方法
            conflict_type = conflict_info.get("conflict_type", "unknown")
            best_fragment_id = conflict_info.get("best_fragment_id")
            fragment_score = conflict_info.get("fragment_score", 0.0)
            
            if conflict_type == "potential_hallucination":
                # 潜在幻觉：使用激活回退
                edited_activations = self._activation_rollback(
                    activations, fragment_fingerprints, best_fragment_id, 
                    layer_id, fragment_score
                )
                self.edit_method_stats["activation_rollback"] += 1
                
            elif conflict_type == "semantic_deviation":
                # 语义偏离：使用重新采样
                edited_activations = self._activation_resampling(
                    activations, fragment_fingerprints, best_fragment_id,
                    layer_id, fragment_score
                )
                self.edit_method_stats["resampling"] += 1
                
            else:
                # 其他情况：使用激活加权
                edited_activations = self._activation_weighting(
                    activations, fragment_fingerprints, fragment_scores,
                    layer_id, conflict_info
                )
                self.edit_method_stats["weighting"] += 1
            
            self.successful_edits += 1
            return edited_activations
            
        except Exception as e:
            logging.error(f"RAG编辑失败: {e}")
            return activations
    
    def _activation_rollback(
        self,
        activations: torch.Tensor,
        fragment_fingerprints: Dict[str, Dict[str, torch.Tensor]],
        fragment_id: str,
        layer_id: str,
        fragment_score: float,
    ) -> torch.Tensor:
        """激活回退：将当前激活向量回退到检索片段的激活状态
        
        Args:
            activations: 原始激活 [batch_size, seq_len, hidden_dim]
            fragment_fingerprints: 片段指纹字典
            fragment_id: 目标片段ID
            layer_id: 当前层ID
            fragment_score: 片段分数
            
        Returns:
            回退后的激活
        """
        if not fragment_id or fragment_id not in fragment_fingerprints:
            return activations
            
        if layer_id not in fragment_fingerprints[fragment_id]:
            return activations
        
        target_activation = fragment_fingerprints[fragment_id][layer_id]
        batch_size, seq_len, hidden_dim = activations.shape
        
        # 计算回退强度（基于片段分数动态调整）
        rollback_strength = self.activation_rollback_strength * fragment_score
        
        # 只对最后一个token进行回退
        edited_activations = activations.clone()
        current_activation = activations[:, -1, :]  # [batch_size, hidden_dim]
        
        # 线性插值回退
        target_expanded = target_activation.unsqueeze(0).expand_as(current_activation)
        rollback_activation = (
            (1 - rollback_strength) * current_activation + 
            rollback_strength * target_expanded
        )
        
        edited_activations[:, -1, :] = rollback_activation
        
        # 记录编辑幅度
        edit_magnitude = torch.norm(rollback_activation - current_activation).item()
        self.edit_magnitudes.append(edit_magnitude)
        
        return edited_activations
    
    def _activation_resampling(
        self,
        activations: torch.Tensor,
        fragment_fingerprints: Dict[str, Dict[str, torch.Tensor]],
        fragment_id: str,
        layer_id: str,
        fragment_score: float,
    ) -> torch.Tensor:
        """激活重新采样：基于检索片段生成新的激活向量
        
        Args:
            activations: 原始激活 [batch_size, seq_len, hidden_dim]
            fragment_fingerprints: 片段指纹字典
            fragment_id: 目标片段ID
            layer_id: 当前层ID
            fragment_score: 片段分数
            
        Returns:
            重新采样后的激活
        """
        if not fragment_id or fragment_id not in fragment_fingerprints:
            return activations
            
        if layer_id not in fragment_fingerprints[fragment_id]:
            return activations
        
        target_activation = fragment_fingerprints[fragment_id][layer_id]
        batch_size, seq_len, hidden_dim = activations.shape
        
        # 基于目标激活和温度参数进行重新采样
        edited_activations = activations.clone()
        current_activation = activations[:, -1, :]  # [batch_size, hidden_dim]
        
        # 计算采样方向和强度
        target_expanded = target_activation.unsqueeze(0).expand_as(current_activation)
        direction = F.normalize(target_expanded - current_activation, p=2, dim=-1)
        
        # 基于温度和片段分数生成噪声
        noise_scale = self.resampling_temperature * (1 - fragment_score)
        noise = torch.randn_like(current_activation) * noise_scale
        
        # 重新采样
        resampled_activation = (
            current_activation + 
            direction * fragment_score * torch.norm(current_activation, p=2, dim=-1, keepdim=True) +
            noise
        )
        
        edited_activations[:, -1, :] = resampled_activation
        
        # 记录编辑幅度
        edit_magnitude = torch.norm(resampled_activation - current_activation).item()
        self.edit_magnitudes.append(edit_magnitude)
        
        return edited_activations
    
    def _activation_weighting(
        self,
        activations: torch.Tensor,
        fragment_fingerprints: Dict[str, Dict[str, torch.Tensor]],
        fragment_scores: Dict[str, float],
        layer_id: str,
        conflict_info: Dict[str, Any],
    ) -> torch.Tensor:
        """激活加权：基于多个检索片段的加权组合
        
        Args:
            activations: 原始激活 [batch_size, seq_len, hidden_dim]
            fragment_fingerprints: 片段指纹字典
            fragment_scores: 片段分数字典
            layer_id: 当前层ID
            conflict_info: 冲突信息
            
        Returns:
            加权后的激活
        """
        batch_size, seq_len, hidden_dim = activations.shape
        edited_activations = activations.clone()
        current_activation = activations[:, -1, :]  # [batch_size, hidden_dim]
        
        # 收集有效的片段激活
        valid_fragments = []
        total_weight = 0.0
        
        for fragment_id, fingerprints in fragment_fingerprints.items():
            if layer_id in fingerprints:
                score = fragment_scores.get(fragment_id, 0.0)
                if score > 0.1:  # 过滤低分片段
                    valid_fragments.append((fingerprints[layer_id], score))
                    total_weight += score
        
        if not valid_fragments or total_weight == 0:
            return activations
        
        # 计算加权激活
        weighted_activation = torch.zeros_like(current_activation)
        
        for fragment_activation, score in valid_fragments:
            weight = score / total_weight
            fragment_expanded = fragment_activation.unsqueeze(0).expand_as(current_activation)
            weighted_activation += weight * fragment_expanded
        
        # 与原始激活进行插值
        weighting_factor = self.activation_weighting_factor * (total_weight / len(valid_fragments))
        final_activation = (
            (1 - weighting_factor) * current_activation +
            weighting_factor * weighted_activation
        )
        
        edited_activations[:, -1, :] = final_activation
        
        # 记录编辑幅度
        edit_magnitude = torch.norm(final_activation - current_activation).item()
        self.edit_magnitudes.append(edit_magnitude)
        
        return edited_activations

    def _compute_error_projection(
        self, error_activation: torch.Tensor, correct_activation: torch.Tensor
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
        projection_coeff = torch.sum(
            error_activation * correct_norm, dim=-1, keepdim=True
        )
        projection = projection_coeff * correct_norm

        return projection

    def _create_edit_mask(
        self,
        batch_size: int,
        seq_len: int,
        is_mc_mode: bool = False,
        prompt_length: Optional[int] = None,
        conflict_position: Optional[int] = None,
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
            mask[:, prompt_length + 1 :] = 1.0
        else:
            # 生成模式：只编辑最后一个token
            mask[:, -1:] = 1.0

        return mask

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_magnitude = np.mean(self.edit_magnitudes) if self.edit_magnitudes else 0.0
        success_rate = self.successful_edits / max(self.edit_count, 1)
        rag_edit_rate = self.rag_edit_count / max(self.edit_count, 1)

        return {
            "edit_count": self.edit_count,
            "successful_edits": self.successful_edits,
            "success_rate": success_rate,
            "average_edit_magnitude": avg_magnitude,
            "layer_edits": dict(self.layer_edits),
            "edit_strength": self.edit_strength,
            "min_confidence": self.min_confidence,
            # RAG编辑统计
            "rag_edit_count": self.rag_edit_count,
            "rag_edit_rate": rag_edit_rate,
            "edit_method_stats": dict(self.edit_method_stats),
            "enable_rag_editing": self.enable_rag_editing,
            "activation_rollback_strength": self.activation_rollback_strength,
            "resampling_temperature": self.resampling_temperature,
            "activation_weighting_factor": self.activation_weighting_factor,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.edit_count = 0
        self.successful_edits = 0
        self.layer_edits.clear()
        self.edit_magnitudes.clear()
        self.rag_edit_count = 0
        self.edit_method_stats.clear()

    def update_edit_strength(self, new_strength: float):
        """
        更新编辑强度

        Args:
            new_strength: 新的编辑强度
        """
        self.edit_strength = new_strength
        # 编辑强度已更新

    def get_edit_magnitude_stats(self) -> Dict[str, float]:
        """
        获取编辑幅度统计

        Returns:
            编辑幅度统计信息
        """
        if not self.edit_magnitudes:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        magnitudes = np.array(self.edit_magnitudes)
        return {
            "mean": float(np.mean(magnitudes)),
            "std": float(np.std(magnitudes)),
            "min": float(np.min(magnitudes)),
            "max": float(np.max(magnitudes)),
            "count": len(magnitudes),
        }
