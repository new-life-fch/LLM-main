"""动态激活指纹构建器
实时为用户问题和检索片段生成激活指纹
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import time

import torch
import torch.nn.functional as F
import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError as e:
    logging.warning(f'导入失败: {e}')
    # TODO: 添加fallback逻辑


class DynamicFingerprintBuilder:
    """动态激活指纹构建器
    
    实时为用户问题和检索片段生成LLM各层的激活指纹，支持批处理和缓存优化
    
    主要功能：
    - 支持用户问题+检索片段的组合指纹构建
    - 支持基于片段ID的缓存机制
    - 支持最后一个token的激活向量提取
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        target_layers: Optional[List[int]] = None,
        device: str = "cuda",
        max_batch_size: int = 16,
        cache_size: int = 500,
        fingerprint_dim: Optional[int] = None,
    ):
        """初始化动态指纹构建器
        
        Args:
            model (torch.nn.Module): LLM模型实例，支持Llama等Transformer模型
            tokenizer (AutoTokenizer): 对应的分词器，用于文本预处理
            target_layers (Optional[List[int]]): 目标层索引列表，None表示使用前10层
            device (str): 计算设备，"cuda"或"cpu"
            max_batch_size (int): 最大批处理大小，用于内存优化
            cache_size (int): 指纹缓存大小，避免重复计算
            fingerprint_dim (Optional[int]): 指纹维度，None表示使用模型隐藏维度
            
        Raises:
            ValueError: 当模型或tokenizer无效时
            RuntimeError: 当设备不可用时
        """
        # 验证输入参数
        if model is None:
            raise ValueError("模型不能为None")
        if tokenizer is None:
            raise ValueError("分词器不能为None")
            
        self.model = model
        self.tokenizer = tokenizer
        
        # 设备配置和验证
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA不可用，回退到CPU")
            device = "cpu"
        self.device = torch.device(device)
        
        # 批处理和缓存配置
        self.max_batch_size = max(1, max_batch_size)  # 确保至少为1
        self.cache_size = max(0, cache_size)  # 确保非负
        
        # 确定模型层数和目标层
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                self.num_layers = len(model.model.layers)
                logging.info(f"检测到模型层数: {self.num_layers}")
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # 支持GPT类模型
                self.num_layers = len(model.transformer.h)
                logging.info(f"检测到GPT类模型层数: {self.num_layers}")
            else:
                self.num_layers = 32  # 默认值
                logging.warning(f"无法检测模型层数，使用默认值: {self.num_layers}")
                
            # 验证和设置目标层
            if target_layers is not None:
                # 验证目标层索引的有效性
                invalid_layers = [l for l in target_layers if l < 0 or l >= self.num_layers]
                if invalid_layers:
                    raise ValueError(f"无效的层索引: {invalid_layers}，模型总层数: {self.num_layers}")
                self.target_layers = target_layers
            else:
                # 默认使用前10层或所有层（取较小值）
                self.target_layers = list(range(min(10, self.num_layers)))
                
            logging.info(f"目标层设置为: {self.target_layers}")
            
        except Exception as e:
            logging.error(f"初始化目标层失败: {e}")
            self.num_layers = 32
            self.target_layers = list(range(10))  # 安全默认值
        
        # 确定指纹维度
        try:
            if fingerprint_dim is None:
                if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                    self.fingerprint_dim = model.config.hidden_size
                    logging.info(f"使用模型隐藏维度: {self.fingerprint_dim}")
                else:
                    self.fingerprint_dim = 4096  # Llama-2-7b默认值
                    logging.warning(f"无法检测模型隐藏维度，使用默认值: {self.fingerprint_dim}")
            else:
                if fingerprint_dim <= 0:
                    raise ValueError(f"指纹维度必须为正数，得到: {fingerprint_dim}")
                self.fingerprint_dim = fingerprint_dim
                logging.info(f"使用指定指纹维度: {self.fingerprint_dim}")
                
        except Exception as e:
            logging.error(f"设置指纹维度失败: {e}")
            self.fingerprint_dim = 4096  # 安全默认值
            
        # 激活钩子管理
        self.activations = {}
        self.hooks = []
        self.hook_registered = False
        self.current_attention_mask = None  # 存储当前批次的attention_mask
        
        # 缓存管理
        self.batch_cache = {}  # batch_key -> fingerprints
        self.rag_fingerprint_cache = {}  # (question, fragment_id) -> fingerprints
        self.fragment_score_cache = {}  # fragment_id -> score
        
        # 统计信息
        self.build_count = 0
        self.cache_hit_count = 0
        self.total_build_time = 0.0
        
        # 动态指纹构建器初始化完成

    def _register_hooks(self):
        """注册激活钩子"""
        if self.hook_registered:
            return
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    # 取第一个输出（通常是hidden states）
                    activation = output[0].detach()
                else:
                    activation = output.detach()
                
                # 使用attention_mask准确定位最后一个有效token的激活
                if activation.dim() == 3:  # [batch, seq_len, hidden_dim]
                    if self.current_attention_mask is not None:
                        # 使用attention_mask找到每个样本的最后一个有效token位置
                        batch_size = activation.size(0)
                        last_token_activations = []
                        
                        for i in range(batch_size):
                            # 找到第i个样本的最后一个有效token位置（attention_mask中最后一个1的位置）
                            mask = self.current_attention_mask[i]
                            last_valid_pos = (mask == 1).nonzero(as_tuple=True)[0][-1].item()
                            last_token_activations.append(activation[i, last_valid_pos, :])
                        
                        activation = torch.stack(last_token_activations, dim=0)  # [batch, hidden_dim]
                    else:
                        # 回退到原来的方法（取最后一个位置）
                        activation = activation[:, -1, :]  # [batch, hidden_dim]
                
                self.activations[name] = activation
            return hook
        
        # 清除之前的钩子
        self._clear_hooks()
        
        # 注册新钩子
        try:
            for layer_idx in self.target_layers:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    if layer_idx < len(self.model.model.layers):
                        # 注册attention层钩子
                        attn_layer = self.model.model.layers[layer_idx].self_attn
                        hook = attn_layer.register_forward_hook(get_activation(f"{layer_idx}.attn"))
                        self.hooks.append(hook)
                        
                        # 注册MLP层钩子
                        mlp_layer = self.model.model.layers[layer_idx].mlp
                        hook = mlp_layer.register_forward_hook(get_activation(f"{layer_idx}.mlp"))
                        self.hooks.append(hook)
            
            self.hook_registered = True
            # 已注册激活钩子
            
        except Exception as e:
            logging.error(f"注册激活钩子失败: {e}")
            self._clear_hooks()

    def _clear_hooks(self):
        """清除激活钩子"""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        self.hooks.clear()
        self.activations.clear()
        self.hook_registered = False  
    
    def _build_fingerprints_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """批量构建激活指纹
        
        Args:
            texts: 文本列表
            
        Returns:
            各层激活指纹字典
        """
        if not texts:
            return {}
        
        # 注册钩子
        self._register_hooks()
        
        layer_fingerprints = {}
        
        try:
            # 分批处理
            for i in range(0, len(texts), self.max_batch_size):
                batch_texts = texts[i:i + self.max_batch_size]
                batch_fingerprints = self._process_text_batch(batch_texts)
                
                # 累积结果
                if not layer_fingerprints:
                    # 初始化
                    for layer_id in batch_fingerprints:
                        layer_fingerprints[layer_id] = batch_fingerprints[layer_id]
                else:
                    # 拼接
                    for layer_id in batch_fingerprints:
                        if layer_id in layer_fingerprints:
                            layer_fingerprints[layer_id] = torch.cat([
                                layer_fingerprints[layer_id],
                                batch_fingerprints[layer_id]
                            ], dim=0)
        
        finally:
            # 清除钩子（可选，保持钩子以提高效率）
            pass
        
        return layer_fingerprints

    def _process_text_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """处理一批文本并生成激活指纹
        
        Args:
            batch_texts (List[str]): 批次文本列表
            
        Returns:
            Dict[str, torch.Tensor]: 批次激活指纹字典
                - key: layer_id (如 "0.attn", "0.mlp")
                - value: 激活张量 [batch_size, hidden_dim]
                
        Raises:
            RuntimeError: 当模型前向传播失败时
        """
        # 输入验证
        if not batch_texts:
            logging.warning("批次文本列表为空")
            return {}
            
        # 过滤空文本
        valid_texts = [text for text in batch_texts if text and text.strip()]
        if not valid_texts:
            logging.warning("批次中没有有效文本")
            return {}
            
        if len(valid_texts) != len(batch_texts):
            logging.warning(f"过滤了 {len(batch_texts) - len(valid_texts)} 个无效文本")
        
        # Tokenization
        try:
            # 动态调整max_length以适应不同的文本长度
            avg_length = sum(len(text.split()) for text in valid_texts) / len(valid_texts)
            max_length = min(512, max(64, int(avg_length * 1.5)))  # 动态长度
            
            # 使用右侧padding（符合llama2训练时的设定）
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            logging.debug(f"Tokenization完成: batch_size={len(valid_texts)}, max_length={max_length}")
            
        except Exception as e:
            logging.error(f"Tokenization失败: {e}")
            logging.error(f"问题文本示例: {valid_texts[0][:100] if valid_texts else 'None'}...")
            return {}
        
        # 清除之前的激活
        self.activations.clear()
        
        # 前向传播
        try:
            # 设置当前批次的attention_mask供钩子函数使用
            self.current_attention_mask = inputs.attention_mask
            
            with torch.no_grad():                
                outputs = self.model(**inputs)
                
            # 验证激活是否被正确收集
            if not self.activations:
                logging.error("未收集到任何激活，可能是钩子注册失败")
                return {}
            
            # 收集和处理激活
            batch_fingerprints = {}
            expected_batch_size = len(valid_texts)
            
            for layer_id, activation in self.activations.items():
                try:
                    if activation.dim() == 2:  # [batch, hidden_dim]
                        # 验证批次大小
                        if activation.size(0) != expected_batch_size:
                            logging.warning(f"层 {layer_id} 批次大小不匹配: 期望 {expected_batch_size}, 实际 {activation.size(0)}")
                            continue
                            
                        # 确保维度匹配
                        if activation.size(1) == self.fingerprint_dim:
                            batch_fingerprints[layer_id] = activation.clone()
                        else:
                            # 维度不匹配时进行调整
                            if activation.size(1) > self.fingerprint_dim:
                                # 截断到目标维度
                                batch_fingerprints[layer_id] = activation[:, :self.fingerprint_dim].clone()
                                logging.debug(f"层 {layer_id} 维度截断: {activation.size(1)} -> {self.fingerprint_dim}")
                            else:
                                # 零填充到目标维度
                                padding = torch.zeros(
                                    activation.size(0), 
                                    self.fingerprint_dim - activation.size(1),
                                    device=self.device,
                                    dtype=activation.dtype
                                )
                                batch_fingerprints[layer_id] = torch.cat([activation, padding], dim=1)
                                logging.debug(f"层 {layer_id} 维度填充: {activation.size(1)} -> {self.fingerprint_dim}")
                    else:
                        logging.warning(f"层 {layer_id} 激活维度异常: {activation.shape}")
                        
                except Exception as layer_e:
                    logging.error(f"处理层 {layer_id} 激活时出错: {layer_e}")
                    continue
            
            if not batch_fingerprints:
                logging.error("未能生成任何有效的批次指纹")
                return {}
                
            logging.debug(f"成功生成 {len(batch_fingerprints)} 个层的批次指纹")
            return batch_fingerprints
            
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"GPU内存不足: {e}")
            logging.error(f"尝试减少批次大小或使用CPU")
            return {}
        except Exception as e:
            logging.error(f"前向传播失败: {e}")
            logging.error(f"输入形状: {inputs.input_ids.shape if hasattr(inputs, 'input_ids') else 'Unknown'}")
            return {}
        finally:
            # 清除attention_mask引用
            self.current_attention_mask = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取构建器统计信息
        
        Returns:
            Dict[str, Any]: 包含性能和缓存统计的字典
        """
        cache_hit_rate = self.cache_hit_count / max(self.build_count, 1)
        avg_build_time = self.total_build_time / max(self.build_count, 1)
        
        return {
            'build_count': self.build_count,
            'cache_hit_count': self.cache_hit_count,
            'cache_hit_rate': round(cache_hit_rate, 4),
            'total_build_time': round(self.total_build_time, 4),
            'avg_build_time': round(avg_build_time, 4),
            'rag_cache_size': len(self.rag_fingerprint_cache),
            'fragment_score_cache_size': len(self.fragment_score_cache),
            'target_layers': self.target_layers,
            'fingerprint_dim': self.fingerprint_dim,
            'max_batch_size': self.max_batch_size,
            'device': str(self.device),
            'hooks_registered': self.hook_registered
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.rag_fingerprint_cache.clear()
        self.batch_cache.clear()
        # 指纹构建器缓存已清空
    
    def build_rag_fingerprints(
        self,
        user_question: str,
        retrieved_fragments: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """为用户问题+检索片段构建多组指纹
        
        这是RAG系统的核心方法，将用户问题与检索片段组合后生成激活指纹，
        用于后续的冲突检测和激活编辑。
        
        Args:
            user_question (str): 用户输入的问题文本
            retrieved_fragments (List[Dict[str, Any]]): 检索到的片段列表，每个片段应包含:
                - text (str): 片段文本内容
                - score (float): 检索相关性分数
                - fragment_id (str, optional): 片段唯一标识符
            use_cache (bool): 是否使用缓存机制，默认True
            
        Returns:
            Dict[str, Dict[str, torch.Tensor]]: 多组指纹字典
                - 外层key: fragment_id
                - 内层key: layer_id (如 "0.attn", "0.mlp")
                - value: 激活指纹张量 [hidden_dim]
                
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当模型前向传播失败时
        """
        # 输入验证
        if not user_question or not user_question.strip():
            raise ValueError("用户问题不能为空")
        if not retrieved_fragments:
            logging.warning("检索片段列表为空，返回空结果")
            return {}
            
        start_time = time.time()
        self.build_count += 1
        
        logging.debug(f"开始构建RAG指纹: 问题长度={len(user_question)}, 片段数量={len(retrieved_fragments)}")
        
        fragment_fingerprints = {}
        uncached_fragments = []
        
        # 检查缓存和预处理片段
        for i, fragment in enumerate(retrieved_fragments):
            # 验证片段格式
            if not isinstance(fragment, dict):
                logging.warning(f"片段 {i} 格式无效，跳过")
                continue
                
            fragment_text = fragment.get('text', '')
            if not fragment_text.strip():
                logging.warning(f"片段 {i} 文本为空，跳过")
                continue
                
            # 生成或获取片段ID
            fragment_id = fragment.get('fragment_id')
            if not fragment_id:
                fragment_id = f"frag_{hash(fragment_text) % 100000000}"  # 生成简短ID
                fragment['fragment_id'] = fragment_id
            
            cache_key = (user_question, fragment_id)
            
            # 仅检查内存缓存
            if use_cache and cache_key in self.rag_fingerprint_cache:
                fragment_fingerprints[fragment_id] = self.rag_fingerprint_cache[cache_key]
                self.cache_hit_count += 1
                logging.debug(f"缓存命中: {fragment_id}")
            else:
                uncached_fragments.append(fragment)
            
            # 缓存片段分数
            if 'score' in fragment:
                self.fragment_score_cache[fragment_id] = fragment['score']
        
        # 处理未缓存的片段
        if uncached_fragments:
            # 构建组合文本：用户问题 + 检索片段
            combined_texts = []
            fragment_ids = []
            
            for fragment in uncached_fragments:
                fragment_text = fragment.get('text', '')
                fragment_id = fragment.get('fragment_id')
                
                # 构建更自然的组合文本（避免人为追加句号导致最后一个token为标点的干扰）
                combined_text = f"{user_question}\n{fragment_text}"
                
                combined_texts.append(combined_text)
                fragment_ids.append(fragment_id)
                
            logging.debug(f"需要构建指纹的片段数量: {len(combined_texts)}")
            
            # 批量构建指纹
            if combined_texts:
                try:
                    batch_fingerprints = self._build_fingerprints_batch(combined_texts)
                    
                    if not batch_fingerprints:
                        logging.error("批量指纹构建失败，返回空结果")
                        return fragment_fingerprints
                    
                    # 分配给各个片段并更新缓存
                    for i, fragment_id in enumerate(fragment_ids):
                        if i >= len(list(batch_fingerprints.values())[0]):  # 检查索引有效性
                            logging.warning(f"片段索引 {i} 超出范围，跳过")
                            continue
                            
                        fragment_fp = {}
                        for layer_id in batch_fingerprints:
                            fragment_fp[layer_id] = batch_fingerprints[layer_id][i].clone()
                        
                        fragment_fingerprints[fragment_id] = fragment_fp
                        
                        # 更新缓存（检查缓存大小限制）
                        cache_key = (user_question, fragment_id)
                        if use_cache and len(self.rag_fingerprint_cache) < self.cache_size:
                            self.rag_fingerprint_cache[cache_key] = fragment_fp
                        
                    logging.debug(f"成功构建 {len(fragment_ids)} 个片段的指纹")
                    
                except Exception as e:
                    logging.error(f"批量构建指纹时发生错误: {e}")
                    # 继续返回已缓存的结果
        
        build_time = time.time() - start_time
        self.total_build_time += build_time
        
        return fragment_fingerprints

    def get_fragment_score(self, fragment_id: str) -> float:
        """获取片段的检索分数
        
        Args:
            fragment_id: 片段ID
            
        Returns:
            片段分数，如果不存在返回0.0
        """
        return self.fragment_score_cache.get(fragment_id, 0.0)
    
    def clear_rag_cache(self):
        """清空RAG相关缓存"""
        self.rag_fingerprint_cache.clear()
        self.fragment_score_cache.clear()
    
    def __del__(self):
        """析构函数，清理钩子"""
        self._clear_hooks()