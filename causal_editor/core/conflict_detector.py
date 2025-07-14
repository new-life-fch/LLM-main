"""
因果冲突检测组件
通过实时激活监测和向量检索来检测因果断裂点
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

from .vector_database import VectorDatabase


class CausalConflictDetector:
    """
    因果冲突检测器
    负责实时监测LLM的激活状态，检测因果断裂点
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        similarity_threshold: float = 0.8,
        conflict_threshold: float = 0.6,
        entity_patterns: Optional[List[str]] = None
    ):
        """
        初始化因果冲突检测器
        
        Args:
            vector_db: 向量数据库实例
            similarity_threshold: 相似度阈值
            conflict_threshold: 冲突判定阈值
            entity_patterns: 实体识别正则表达式模式
        """
        self.vector_db = vector_db
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold
        
        # 实体识别模式
        self.entity_patterns = entity_patterns or [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 人名、地名等
            r'\b\d{4}\b',  # 年份
            r'\b\d+(?:\.\d+)?\b',  # 数字
            r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b'  # 专有名词
        ]
        
        # 统计信息
        self.detection_count = 0
        self.conflict_count = 0
        self.layer_conflicts = defaultdict(int)
        
        logging.info("因果冲突检测器初始化完成")
    
    def detect_conflict(
        self,
        activations: torch.Tensor,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
        layer_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        检测因果冲突
        
        Args:
            activations: 当前激活状态 [batch_size, seq_len, hidden_dim]
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens  
            layer_id: 当前层ID
            
        Returns:
            冲突信息字典
        """
        self.detection_count += 1
        
        conflict_info = {
            'has_conflict': False,
            'conflict_type': None,
            'conflict_position': None,
            'retrieved_knowledge': [],
            'confidence': 0.0,
            'layer_id': layer_id
        }
        
        try:
            # 提取当前激活向量（通常关注最后一个token的激活）
            batch_size, seq_len, hidden_dim = activations.shape
            current_activation = activations[:, -1, :].squeeze(0)  # [hidden_dim]
            
            # 检索相似的知识激活
            retrieved_knowledge = self.vector_db.search(
                query_vector=current_activation,
                layer_id=layer_id,
                k=20,
                score_threshold=0.3
            )
            
            if not retrieved_knowledge:
                return conflict_info
            
            # 分析生成的tokens和检索到的知识
            conflict_result = self._analyze_conflict(
                generated_tokens=generated_tokens,
                context_tokens=context_tokens,
                retrieved_knowledge=retrieved_knowledge,
                current_activation=current_activation
            )
            
            if conflict_result['has_conflict']:
                self.conflict_count += 1
                self.layer_conflicts[layer_id] += 1
                
                conflict_info.update(conflict_result)
                conflict_info['retrieved_knowledge'] = retrieved_knowledge
                
                logging.debug(f"检测到冲突 - 层: {layer_id}, 类型: {conflict_result['conflict_type']}")
            
            return conflict_info
            
        except Exception as e:
            logging.error(f"冲突检测失败: {e}")
            return conflict_info
    
    def _analyze_conflict(
        self,
        generated_tokens: Optional[List[str]],
        context_tokens: Optional[List[str]],
        retrieved_knowledge: List[Dict[str, Any]],
        current_activation: torch.Tensor
    ) -> Dict[str, Any]:
        """
        分析是否存在冲突
        
        Args:
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            retrieved_knowledge: 检索到的知识
            current_activation: 当前激活向量
            
        Returns:
            冲突分析结果
        """
        conflict_result = {
            'has_conflict': False,
            'conflict_type': None,
            'conflict_position': None,
            'confidence': 0.0,
            'correct_object': None,
            'incorrect_object': None
        }
        
        if not generated_tokens or not retrieved_knowledge:
            return conflict_result
        
        # 获取最近生成的token
        recent_token = generated_tokens[-1] if generated_tokens else ""
        
        # 检查是否是关键实体或数字
        if not self._is_key_entity(recent_token):
            return conflict_result
        
        # 分析检索到的知识
        knowledge_analysis = self._analyze_retrieved_knowledge(retrieved_knowledge)
        
        if not knowledge_analysis['has_consensus']:
            return conflict_result
        
        # 检查生成的token与知识库中的信息是否冲突
        consensus_object = knowledge_analysis['consensus_object']
        confidence = knowledge_analysis['confidence']
        
        if self._tokens_conflict(recent_token, consensus_object):
            conflict_result = {
                'has_conflict': True,
                'conflict_type': 'factual_error',
                'conflict_position': len(generated_tokens) - 1,
                'confidence': confidence,
                'correct_object': consensus_object,
                'incorrect_object': recent_token
            }
        
        return conflict_result
    
    def _is_key_entity(self, token: str) -> bool:
        """
        判断token是否为关键实体
        
        Args:
            token: 待检查的token
            
        Returns:
            是否为关键实体
        """
        if not token or len(token) < 2:
            return False
        
        # 检查是否匹配实体模式
        for pattern in self.entity_patterns:
            if re.match(pattern, token):
                return True
        
        return False
    
    def _analyze_retrieved_knowledge(self, retrieved_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析检索到的知识，寻找共识
        
        Args:
            retrieved_knowledge: 检索到的知识列表
            
        Returns:
            知识分析结果
        """
        analysis = {
            'has_consensus': False,
            'consensus_object': None,
            'confidence': 0.0,
            'supporting_count': 0
        }
        
        if not retrieved_knowledge:
            return analysis
        
        # 统计不同object的出现频次和置信度
        object_votes = defaultdict(list)
        
        for knowledge in retrieved_knowledge:
            obj = knowledge.get('object', '').strip()
            if obj:
                score = knowledge.get('similarity_score', 0.0)
                confidence = knowledge.get('confidence', 1.0)
                combined_score = score * confidence
                object_votes[obj].append(combined_score)
        
        if not object_votes:
            return analysis
        
        # 计算每个object的综合得分
        object_scores = {}
        for obj, scores in object_votes.items():
            object_scores[obj] = {
                'avg_score': np.mean(scores),
                'count': len(scores),
                'total_score': sum(scores)
            }
        
        # 找到得分最高的object
        best_object = max(object_scores.keys(), 
                         key=lambda x: object_scores[x]['total_score'])
        
        best_info = object_scores[best_object]
        
        # 判断是否有足够的共识
        if (best_info['count'] >= 3 and 
            best_info['avg_score'] > self.similarity_threshold):
            analysis = {
                'has_consensus': True,
                'consensus_object': best_object,
                'confidence': min(best_info['avg_score'], 1.0),
                'supporting_count': best_info['count']
            }
        
        return analysis
    
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
                return abs(gen_num - correct_num) > 0.1 * max(abs(gen_num), abs(correct_num))
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
                'layer_id': layer_id,
                'conflict_count': self.layer_conflicts.get(layer_id, 0),
                'detection_count': self.detection_count
            }
        else:
            return {
                'total_detections': self.detection_count,
                'total_conflicts': self.conflict_count,
                'conflict_rate': self.conflict_count / max(self.detection_count, 1),
                'layer_conflicts': dict(self.layer_conflicts)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'detection_count': self.detection_count,
            'conflict_count': self.conflict_count,
            'conflict_rate': self.conflict_count / max(self.detection_count, 1),
            'layer_conflicts': dict(self.layer_conflicts),
            'similarity_threshold': self.similarity_threshold,
            'conflict_threshold': self.conflict_threshold
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.detection_count = 0
        self.conflict_count = 0
        self.layer_conflicts.clear()
    
    def update_thresholds(self, similarity_threshold: Optional[float] = None, 
                         conflict_threshold: Optional[float] = None):
        """
        更新阈值参数
        
        Args:
            similarity_threshold: 新的相似度阈值
            conflict_threshold: 新的冲突阈值
        """
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            logging.info(f"相似度阈值更新为: {similarity_threshold}")
        
        if conflict_threshold is not None:
            self.conflict_threshold = conflict_threshold
            logging.info(f"冲突阈值更新为: {conflict_threshold}") 