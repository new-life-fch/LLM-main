"""因果冲突检测组件
通过实时激活监测和动态指纹比对来检测因果断裂点
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np



class CausalConflictDetector:
    """
    因果冲突检测器
    负责实时监测LLM的激活状态，检测因果断裂点
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        conflict_threshold: float = 0.6,
        enable_dynamic_threshold: bool = True,
        threshold_adjustment_factor: float = 0.3,
    ):
        """
        初始化因果冲突检测器

        Args:
            similarity_threshold: 相似度阈值，用于判断激活向量的相似程度 (0.0-1.0)
            conflict_threshold: 冲突判定阈值，低于此值认为存在冲突 (0.0-1.0)
            enable_dynamic_threshold: 是否启用基于片段分数的动态阈值调整
            threshold_adjustment_factor: 阈值调整因子，控制动态调整的幅度 (0.0-1.0)
        """
        # 参数验证
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold必须在[0.0, 1.0]范围内，当前值: {similarity_threshold}")
        if not (0.0 <= conflict_threshold <= 1.0):
            raise ValueError(f"conflict_threshold必须在[0.0, 1.0]范围内，当前值: {conflict_threshold}")
        if conflict_threshold >= similarity_threshold:
            raise ValueError(f"conflict_threshold ({conflict_threshold}) 必须小于 similarity_threshold ({similarity_threshold})")
        if not (0.0 <= threshold_adjustment_factor <= 1.0):
            raise ValueError(f"threshold_adjustment_factor必须在[0.0, 1.0]范围内，当前值: {threshold_adjustment_factor}")
        
        # 基础阈值参数
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold
        # 保存基础阈值用于动态调整
        self.base_similarity_threshold = similarity_threshold
        self.base_conflict_threshold = conflict_threshold
        
        # 动态阈值相关参数
        self.enable_dynamic_threshold = enable_dynamic_threshold
        self.threshold_adjustment_factor = threshold_adjustment_factor

        # 统计信息
        self.detection_count = 0  # 总检测次数
        self.conflict_count = 0   # 检测到的冲突次数
        self.layer_conflicts = defaultdict(int)  # 各层的冲突统计

        logging.info(f"因果冲突检测器初始化完成 - 相似度阈值: {similarity_threshold}, 冲突阈值: {conflict_threshold}, 动态阈值: {enable_dynamic_threshold}")

    def detect_conflict(
        self,
        activations: torch.Tensor,
        correction_vector: Optional[torch.Tensor] = None,
        confidence: float = 0.0,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
        layer_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        检测因果冲突（动态模式）
        
        基于传入的激活状态、修正向量和置信度来判断是否存在因果冲突。
        这是一个通用的冲突检测接口，适用于各种冲突检测场景。

        Args:
            activations: 当前激活状态 [batch_size, seq_len, hidden_dim] 或 [hidden_dim]
            correction_vector: 修正向量（来自动态指纹），如果提供则用于计算修正强度
            confidence: 置信度分数 (0.0-1.0)，通常来自检索系统或相似度计算
            generated_tokens: 已生成的tokens列表，用于实体冲突检测
            context_tokens: 上下文tokens列表，用于上下文一致性检查
            layer_id: 当前层ID，用于统计和调试

        Returns:
            冲突信息字典，包含冲突状态、类型、置信度等信息
        """
        self.detection_count += 1

        # 初始化冲突信息结构
        conflict_info = {
            "has_conflict": False,
            "conflict_type": None,
            "conflict_position": None,
            "confidence": confidence,
            "layer_id": layer_id,
            "correction_vector": correction_vector,
            "correction_strength": 0.0,
            "entity_conflicts": [],
        }

        try:
            # 输入验证
            if activations is None:
                logging.warning(f"层 {layer_id}: 激活状态为空，跳过冲突检测")
                return conflict_info
            
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logging.warning(f"层 {layer_id}: 置信度值无效 ({confidence})，使用默认值 0.0")
                confidence = 0.0
                conflict_info["confidence"] = confidence

            # 动态模式：基于传入的置信度和修正向量判断冲突
            if correction_vector is not None and confidence > self.conflict_threshold:
                # 计算修正强度
                if activations.dim() > 1:
                    # 如果是多维激活，取最后一个token的激活
                    current_activation = activations[:, -1, :] if activations.dim() == 3 else activations[-1, :]
                else:
                    current_activation = activations
                
                correction_strength = self.calculate_correction_strength(
                    current_activation, correction_vector
                )
                
                self.conflict_count += 1
                self.layer_conflicts[layer_id] += 1

                conflict_info.update({
                    "has_conflict": True,
                    "conflict_type": "dynamic_knowledge_inconsistency",
                    "confidence": confidence,
                    "correction_strength": correction_strength,
                })

                logging.debug(f"层 {layer_id}: 检测到动态冲突 - 置信度: {confidence:.3f}, 修正强度: {correction_strength:.3f}")
            
            # 实体级冲突检测（如果提供了tokens）
            if generated_tokens and context_tokens:
                entity_conflicts = self._detect_entity_conflicts(generated_tokens, context_tokens)
                if entity_conflicts:
                    conflict_info["entity_conflicts"] = entity_conflicts
                    if not conflict_info["has_conflict"]:
                        conflict_info.update({
                            "has_conflict": True,
                            "conflict_type": "entity_inconsistency",
                            "confidence": max(0.5, confidence),  # 实体冲突给予中等置信度
                        })
                        self.conflict_count += 1
                        self.layer_conflicts[layer_id] += 1

            return conflict_info

        except Exception as e:
            logging.error(f"层 {layer_id}: 动态冲突检测失败 - {str(e)}")
            return conflict_info

    def analyze_dynamic_conflict(
        self,
        current_activation: torch.Tensor,
        correction_vector: torch.Tensor,
        similarity_score: float,
        entity_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析动态冲突

        Args:
            current_activation: 当前激活向量
            correction_vector: 修正向量
            similarity_score: 相似度分数
            entity_info: 实体信息

        Returns:
            冲突分析结果
        """
        conflict_result = {
            "has_conflict": False,
            "conflict_type": None,
            "confidence": similarity_score,
            "correction_strength": 0.0,
        }

        try:
            # 计算修正强度
            correction_norm = torch.norm(correction_vector).item()
            activation_norm = torch.norm(current_activation).item()
            
            if activation_norm > 0:
                correction_strength = correction_norm / activation_norm
            else:
                correction_strength = 0.0

            # 判断是否需要修正
            if (similarity_score > self.similarity_threshold and 
                correction_strength > 0.1):  # 修正强度阈值
                
                conflict_result = {
                    "has_conflict": True,
                    "conflict_type": "dynamic_knowledge_inconsistency",
                    "confidence": similarity_score,
                    "correction_strength": correction_strength,
                    "entity_info": entity_info,
                }

            return conflict_result

        except Exception as e:
            logging.error(f"动态冲突分析失败: {e}")
            return conflict_result

    def _detect_entity_conflicts(
        self, 
        generated_tokens: List[str], 
        context_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        检测实体级别的冲突
        
        Args:
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            
        Returns:
            实体冲突列表
        """
        entity_conflicts = []
        
        try:
            # 提取生成文本中的关键实体
            generated_entities = [token for token in generated_tokens if self._is_key_entity(token)]
            context_entities = [token for token in context_tokens if self._is_key_entity(token)]
            
            # 检查实体冲突
            for gen_entity in generated_entities:
                for ctx_entity in context_entities:
                    if self._tokens_conflict(gen_entity, ctx_entity):
                        entity_conflicts.append({
                            "generated_entity": gen_entity,
                            "context_entity": ctx_entity,
                            "conflict_type": "entity_mismatch"
                        })
            
            return entity_conflicts
            
        except Exception as e:
            logging.error(f"实体冲突检测失败: {e}")
            return []
    
    def _is_key_entity(self, token: str) -> bool:
        """
        判断token是否为关键实体
        
        使用预定义的正则表达式模式来识别关键实体，如人名、地名、数字等。

        Args:
            token: 待检查的token字符串

        Returns:
            bool: 如果token匹配任何实体模式则返回True，否则返回False
        """
        if not token or len(token) < 2:
            return False

        # 检查是否匹配实体模式
        for pattern in self.entity_patterns:
            if re.match(pattern, token):
                return True

        return False

    def calculate_correction_strength(
        self, 
        current_activation: torch.Tensor, 
        correction_vector: torch.Tensor
    ) -> float:
        """
        计算修正强度

        Args:
            current_activation: 当前激活向量
            correction_vector: 修正向量

        Returns:
            修正强度
        """
        try:
            # 计算向量间的角度差异
            cosine_sim = F.cosine_similarity(
                current_activation.unsqueeze(0),
                correction_vector.unsqueeze(0)
            ).item()
            
            # 计算相对强度
            correction_norm = torch.norm(correction_vector).item()
            activation_norm = torch.norm(current_activation).item()
            
            if activation_norm > 0:
                relative_strength = correction_norm / activation_norm
            else:
                relative_strength = 0.0
            
            # 综合考虑角度和强度
            correction_strength = (1 - cosine_sim) * relative_strength
            
            return correction_strength
            
        except Exception as e:
            logging.error(f"计算修正强度失败: {e}")
            return 0.0

    def _tokens_conflict(self, generated_token: str, correct_object: str) -> bool:
        """
        判断生成的token与正确答案是否冲突

        Args:
            generated_token: 生成的token
            correct_object: 正确的object

        Returns:
            是否冲突
        """
        if not generated_token or not correct_object:
            return False

        # 标准化比较
        gen_normalized = generated_token.lower().strip()
        correct_normalized = correct_object.lower().strip()

        # 完全匹配
        if gen_normalized == correct_normalized:
            return False

        # 部分匹配检查（对于复合词）
        if gen_normalized in correct_normalized or correct_normalized in gen_normalized:
            return False

        # 数字冲突检查
        if self._is_numeric(generated_token) and self._is_numeric(correct_object):
            try:
                gen_num = float(generated_token)
                correct_num = float(correct_object)
                # 如果数字差异显著，认为是冲突
                return abs(gen_num - correct_num) > 0.1 * max(
                    abs(gen_num), abs(correct_num)
                )
            except ValueError:
                pass

        # 默认认为是冲突（不同的实体名称）
        return True

    def _is_numeric(self, text: str) -> bool:
        """判断文本是否为数字"""
        try:
            float(text)
            return True
        except ValueError:
            return False

    def get_conflict_patterns(self, layer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取冲突模式分析

        Args:
            layer_id: 特定层ID，None表示所有层

        Returns:
            冲突模式统计
        """
        if layer_id:
            return {
                "layer_id": layer_id,
                "conflict_count": self.layer_conflicts.get(layer_id, 0),
                "detection_count": self.detection_count,
            }
        else:
            return {
                "total_detections": self.detection_count,
                "total_conflicts": self.conflict_count,
                "conflict_rate": self.conflict_count / max(self.detection_count, 1),
                "layer_conflicts": dict(self.layer_conflicts),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "detection_count": self.detection_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_count / max(self.detection_count, 1),
            "layer_conflicts": dict(self.layer_conflicts),
            "similarity_threshold": self.similarity_threshold,
            "conflict_threshold": self.conflict_threshold,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.detection_count = 0
        self.conflict_count = 0
        self.layer_conflicts.clear()

    def update_thresholds(
        self,
        similarity_threshold: Optional[float] = None,
        conflict_threshold: Optional[float] = None,
    ):
        """
        更新阈值参数
        
        提供运行时动态调整检测阈值的能力，支持根据实际使用情况优化检测效果。

        Args:
            similarity_threshold: 新的相似度阈值 (0.0-1.0)，None表示不更新
            conflict_threshold: 新的冲突阈值 (0.0-1.0)，None表示不更新
            
        Raises:
            ValueError: 当阈值参数不在有效范围内时
        """
        old_sim_threshold = self.similarity_threshold
        old_conf_threshold = self.conflict_threshold
        
        try:
            if similarity_threshold is not None:
                if not (0.0 <= similarity_threshold <= 1.0):
                    raise ValueError(f"similarity_threshold必须在[0.0, 1.0]范围内，当前值: {similarity_threshold}")
                self.similarity_threshold = similarity_threshold
                logging.info(f"相似度阈值已更新: {old_sim_threshold:.3f} -> {similarity_threshold:.3f}")

            if conflict_threshold is not None:
                if not (0.0 <= conflict_threshold <= 1.0):
                    raise ValueError(f"conflict_threshold必须在[0.0, 1.0]范围内，当前值: {conflict_threshold}")
                if conflict_threshold >= self.similarity_threshold:
                    raise ValueError(f"conflict_threshold ({conflict_threshold}) 必须小于 similarity_threshold ({self.similarity_threshold})")
                self.conflict_threshold = conflict_threshold
                logging.info(f"冲突阈值已更新: {old_conf_threshold:.3f} -> {conflict_threshold:.3f}")
                
            # 同时更新基础阈值（用于动态调整）
            if similarity_threshold is not None:
                self.base_similarity_threshold = similarity_threshold
            if conflict_threshold is not None:
                self.base_conflict_threshold = conflict_threshold
                
        except ValueError as e:
            logging.error(f"阈值更新失败: {e}")
            raise
    
    def detect_rag_conflict(
        self,
        current_activation: torch.Tensor,
        fragment_fingerprints: Dict[str, Dict[str, torch.Tensor]],
        fragment_scores: Dict[str, float],
        layer_id: str = "unknown",
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """检测RAG模式下的因果冲突
        
        这是RAG系统的核心冲突检测方法，通过比较当前激活与检索片段的指纹
        来判断是否存在知识冲突，支持动态阈值调整以提高检测精度。
        
        Args:
            current_activation: 当前激活向量 [hidden_dim]，来自模型的实时激活
            fragment_fingerprints: 片段指纹字典 {fragment_id: {layer_id: tensor}}，预构建的知识片段指纹
            fragment_scores: 片段检索分数字典 {fragment_id: score}，检索系统给出的相关性分数
            layer_id: 当前层ID，用于定位具体的模型层
            generated_tokens: 已生成的tokens，用于实体级冲突检测
            context_tokens: 上下文tokens，用于上下文一致性检查
            
        Returns:
            Dict[str, Any]: 详细的冲突检测结果，包含冲突状态、类型、置信度等信息
        """
        self.detection_count += 1
        
        # 初始化冲突信息结构
        conflict_info = {
            "has_conflict": False,
            "conflict_type": None,
            "confidence": 0.0,
            "layer_id": layer_id,
            "best_fragment_id": None,
            "max_similarity": 0.0,
            "fragment_score": 0.0,
            "dynamic_similarity_threshold": self.similarity_threshold,
            "dynamic_conflict_threshold": self.conflict_threshold,
            "similarity_scores": {},  # 记录所有片段的相似度
            "processing_time": 0.0,
        }
        
        # 输入验证
        if not fragment_fingerprints:
            logging.debug(f"层 {layer_id}: 没有可用的片段指纹，跳过RAG冲突检测")
            return conflict_info
            
        if current_activation is None or current_activation.numel() == 0:
            logging.warning(f"层 {layer_id}: 当前激活为空，跳过RAG冲突检测")
            return conflict_info
        
        import time
        start_time = time.time()
        
        try:
            # 计算与所有片段指纹的相似度
            max_similarity = 0.0
            best_fragment_id = None
            best_fragment_score = 0.0
            similarity_scores = {}
            
            for fragment_id, fingerprints in fragment_fingerprints.items():
                if layer_id in fingerprints:
                    fragment_activation = fingerprints[layer_id]
                    
                    # 确保张量维度匹配
                    if current_activation.shape != fragment_activation.shape:
                        logging.warning(f"层 {layer_id}: 激活维度不匹配 - 当前: {current_activation.shape}, 片段: {fragment_activation.shape}")
                        continue
                    
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(
                        current_activation.unsqueeze(0),
                        fragment_activation.unsqueeze(0)
                    ).item()
                    
                    similarity_scores[fragment_id] = similarity
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_fragment_id = fragment_id
                        best_fragment_score = fragment_scores.get(fragment_id, 0.0)
            
            # 动态调整阈值
            if self.enable_dynamic_threshold and best_fragment_id is not None:
                dynamic_thresholds = self._calculate_dynamic_thresholds(best_fragment_score)
                conflict_info["dynamic_similarity_threshold"] = dynamic_thresholds["similarity"]
                conflict_info["dynamic_conflict_threshold"] = dynamic_thresholds["conflict"]
            else:
                dynamic_thresholds = {
                    "similarity": self.similarity_threshold,
                    "conflict": self.conflict_threshold
                }
            
            # 更新冲突信息
            conflict_info.update({
                "max_similarity": max_similarity,
                "best_fragment_id": best_fragment_id,
                "fragment_score": best_fragment_score,
                "similarity_scores": similarity_scores,
                "processing_time": time.time() - start_time,
            })
            
            # 判断冲突类型 - 使用更精细的判断逻辑
            if max_similarity < dynamic_thresholds["conflict"] and best_fragment_score > 0.7:
                # 检索片段分数很高但相似度很低，可能产生幻觉
                confidence = best_fragment_score * (1 - max_similarity)
                conflict_info.update({
                    "has_conflict": True,
                    "conflict_type": "potential_hallucination",
                    "confidence": confidence,
                })
                self.conflict_count += 1
                self.layer_conflicts[layer_id] += 1
                logging.debug(f"层 {layer_id}: 检测到潜在幻觉 - 相似度: {max_similarity:.3f}, 片段分数: {best_fragment_score:.3f}")
                
            elif (dynamic_thresholds["conflict"] <= max_similarity < dynamic_thresholds["similarity"]):
                # 相似度中等，可能产生语义偏离
                confidence = (dynamic_thresholds["similarity"] - max_similarity) * best_fragment_score
                conflict_info.update({
                    "has_conflict": True,
                    "conflict_type": "semantic_deviation",
                    "confidence": confidence,
                })
                self.conflict_count += 1
                self.layer_conflicts[layer_id] += 1
                logging.debug(f"层 {layer_id}: 检测到语义偏离 - 相似度: {max_similarity:.3f}, 阈值范围: [{dynamic_thresholds['conflict']:.3f}, {dynamic_thresholds['similarity']:.3f}]")
                
            elif max_similarity >= dynamic_thresholds["similarity"]:
                # 相似度很高，不需要编辑
                confidence = max_similarity * best_fragment_score
                conflict_info.update({
                    "has_conflict": False,
                    "conflict_type": "no_conflict",
                    "confidence": confidence,
                })
                logging.debug(f"层 {layer_id}: 无冲突 - 相似度: {max_similarity:.3f} >= 阈值: {dynamic_thresholds['similarity']:.3f}")
            
            else:
                # 其他情况：相似度很低且片段分数也不高
                conflict_info.update({
                    "has_conflict": False,
                    "conflict_type": "insufficient_evidence",
                    "confidence": max_similarity * best_fragment_score,
                })
            
            # 额外的实体级冲突检测
            if generated_tokens and context_tokens:
                entity_conflicts = self._detect_entity_conflicts(generated_tokens, context_tokens)
                if entity_conflicts:
                    conflict_info["entity_conflicts"] = entity_conflicts
                    # 如果没有其他冲突但有实体冲突，标记为实体不一致
                    if not conflict_info["has_conflict"]:
                        conflict_info.update({
                            "has_conflict": True,
                            "conflict_type": "entity_inconsistency",
                            "confidence": max(0.6, conflict_info["confidence"]),
                        })
                        self.conflict_count += 1
                        self.layer_conflicts[layer_id] += 1
            
            return conflict_info
            
        except Exception as e:
            logging.error(f"层 {layer_id}: RAG冲突检测失败 - {str(e)}")
            conflict_info["processing_time"] = time.time() - start_time
            return conflict_info
    
    def _calculate_dynamic_thresholds(self, fragment_score: float) -> Dict[str, float]:
        """根据片段分数动态计算阈值
        
        Args:
            fragment_score: 片段检索分数 (0.0-1.0)
            
        Returns:
            动态阈值字典 {"similarity": float, "conflict": float}
        """
        # 片段分数越高，阈值越低（更容易触发编辑）
        # 片段分数越低，阈值越高（更难触发编辑）
        
        # 计算调整量
        score_factor = max(0.0, min(1.0, fragment_score))  # 确保在[0,1]范围内
        adjustment = self.threshold_adjustment_factor * (1.0 - score_factor)
        
        # 动态相似度阈值
        dynamic_similarity_threshold = min(
            1.0, 
            self.base_similarity_threshold + adjustment
        )
        
        # 动态冲突阈值
        dynamic_conflict_threshold = min(
            dynamic_similarity_threshold - 0.1,  # 确保冲突阈值小于相似度阈值
            self.base_conflict_threshold + adjustment
        )
        
        return {
            "similarity": dynamic_similarity_threshold,
            "conflict": max(0.1, dynamic_conflict_threshold)  # 确保最小阈值
        }
