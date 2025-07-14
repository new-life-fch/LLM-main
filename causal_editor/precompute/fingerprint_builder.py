"""
激活指纹构建组件
为知识三元组生成LLM的激活指纹
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationFingerprintBuilder:
    """
    激活指纹构建器
    为给定的知识三元组生成LLM各层的激活指纹
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
        target_layers: Optional[List[int]] = None,
        cache_dir: str = "./cache/fingerprints"
    ):
        """
        初始化激活指纹构建器
        
        Args:
            model_name: 模型名称或路径
            device: 计算设备
            batch_size: 批处理大小
            target_layers: 目标层列表，None表示所有层
            cache_dir: 缓存目录
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型和tokenizer
        logging.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 确定目标层
        self.num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 32
        self.target_layers = target_layers or list(range(self.num_layers))
        
        # 激活钩子
        self.activations = {}
        self.hooks = []
        
        logging.info(f"激活指纹构建器初始化完成，目标层: {self.target_layers}")
    
    def _register_hooks(self):
        """注册激活钩子"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # 清除之前的钩子
        self._clear_hooks()
        
        # 注册新钩子
        for layer_idx in self.target_layers:
            # 注册attention层钩子
            if hasattr(self.model.model, 'layers'):
                attn_layer = self.model.model.layers[layer_idx].self_attn
                hook = attn_layer.register_forward_hook(get_activation(f"{layer_idx}.attn"))
                self.hooks.append(hook)
                
                # 注册MLP层钩子
                mlp_layer = self.model.model.layers[layer_idx].mlp
                hook = mlp_layer.register_forward_hook(get_activation(f"{layer_idx}.mlp"))
                self.hooks.append(hook)
    
    def _clear_hooks(self):
        """清除激活钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def build_fingerprints(
        self,
        knowledge_triplets: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        为知识三元组构建激活指纹
        
        Args:
            knowledge_triplets: 知识三元组列表
            output_path: 输出路径，None表示不保存
            
        Returns:
            各层激活指纹字典 {layer_id: [fingerprints]}
        """
        # 检查缓存
        if output_path and Path(output_path).exists():
            logging.info(f"从缓存加载激活指纹: {output_path}")
            return self._load_fingerprints(output_path)
        
        logging.info(f"开始构建 {len(knowledge_triplets)} 个三元组的激活指纹")
        
        # 注册激活钩子
        self._register_hooks()
        
        layer_fingerprints = {f"{layer_idx}.attn": [] for layer_idx in self.target_layers}
        layer_fingerprints.update({f"{layer_idx}.mlp": [] for layer_idx in self.target_layers})
        
        try:
            # 分批处理
            for i in tqdm(range(0, len(knowledge_triplets), self.batch_size), 
                         desc="构建激活指纹"):
                batch_triplets = knowledge_triplets[i:i + self.batch_size]
                batch_fingerprints = self._process_batch(batch_triplets)
                
                # 累积结果
                for layer_id, fingerprints in batch_fingerprints.items():
                    layer_fingerprints[layer_id].extend(fingerprints)
        
        finally:
            # 清除钩子
            self._clear_hooks()
        
        # 转换为numpy数组
        for layer_id in layer_fingerprints:
            if layer_fingerprints[layer_id]:
                layer_fingerprints[layer_id] = np.stack(layer_fingerprints[layer_id])
            else:
                layer_fingerprints[layer_id] = np.array([])
        
        # 保存结果
        if output_path:
            self._save_fingerprints(layer_fingerprints, output_path, knowledge_triplets)
        
        logging.info("激活指纹构建完成")
        return layer_fingerprints
    
    def _process_batch(self, batch_triplets: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
        """
        处理一批三元组
        
        Args:
            batch_triplets: 批次三元组
            
        Returns:
            批次激活指纹
        """
        # 准备输入文本
        texts = []
        for triplet in batch_triplets:
            text = triplet.get('text', '')
            if not text:
                # 如果没有预生成的文本，创建一个
                text = f"{triplet['subject']} {triplet.get('relation_label', triplet['relation'])} {triplet['object']}."
            texts.append(text)
        
        # Tokenization
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # 清除之前的激活
        self.activations.clear()
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # 提取激活指纹
        batch_fingerprints = {}
        
        for layer_id, activation in self.activations.items():
            # 提取每个样本在特定位置的激活
            fingerprints = []
            
            for i in range(len(batch_triplets)):
                # 获取该样本的序列长度（排除padding）
                seq_len = inputs['attention_mask'][i].sum().item()
                
                # 提取关键位置的激活
                key_positions = self._get_key_positions(
                    texts[i], 
                    batch_triplets[i],
                    inputs['input_ids'][i],
                    seq_len
                )
                
                # 平均关键位置的激活作为指纹
                sample_activation = activation[i]  # [seq_len, hidden_dim]
                
                if key_positions:
                    key_activations = sample_activation[key_positions]
                    fingerprint = key_activations.mean(dim=0)
                else:
                    # 如果没找到关键位置，使用最后一个非padding位置
                    fingerprint = sample_activation[seq_len - 1]
                
                fingerprints.append(fingerprint.cpu().numpy())
            
            batch_fingerprints[layer_id] = fingerprints
        
        return batch_fingerprints
    
    def _get_key_positions(
        self,
        text: str,
        triplet: Dict[str, Any],
        input_ids: torch.Tensor,
        seq_len: int
    ) -> List[int]:
        """
        获取关键位置（实体和对象token的位置）
        
        Args:
            text: 输入文本
            triplet: 知识三元组
            input_ids: token IDs
            seq_len: 序列长度
            
        Returns:
            关键位置列表
        """
        key_positions = []
        
        # 尝试找到subject和object在序列中的位置
        subject = triplet['subject']
        obj = triplet['object']
        
        # Tokenize subject和object
        subject_tokens = self.tokenizer.encode(subject, add_special_tokens=False)
        object_tokens = self.tokenizer.encode(obj, add_special_tokens=False)
        
        # 在input_ids中查找匹配位置
        input_ids_list = input_ids[:seq_len].tolist()
        
        # 查找subject位置
        subject_pos = self._find_token_sequence(input_ids_list, subject_tokens)
        if subject_pos is not None:
            key_positions.extend(range(subject_pos, subject_pos + len(subject_tokens)))
        
        # 查找object位置
        object_pos = self._find_token_sequence(input_ids_list, object_tokens)
        if object_pos is not None:
            key_positions.extend(range(object_pos, object_pos + len(object_tokens)))
        
        # 如果找不到关键位置，使用最后几个位置
        if not key_positions:
            key_positions = list(range(max(0, seq_len - 3), seq_len))
        
        return key_positions
    
    def _find_token_sequence(self, input_ids: List[int], target_tokens: List[int]) -> Optional[int]:
        """
        在token序列中查找目标子序列
        
        Args:
            input_ids: 输入token序列
            target_tokens: 目标token序列
            
        Returns:
            找到的起始位置，None表示未找到
        """
        if not target_tokens:
            return None
        
        for i in range(len(input_ids) - len(target_tokens) + 1):
            if input_ids[i:i + len(target_tokens)] == target_tokens:
                return i
        
        return None
    
    def _save_fingerprints(
        self,
        layer_fingerprints: Dict[str, np.ndarray],
        output_path: str,
        knowledge_triplets: List[Dict[str, Any]]
    ):
        """
        保存激活指纹
        
        Args:
            layer_fingerprints: 层级激活指纹
            output_path: 输出路径
            knowledge_triplets: 对应的知识三元组
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存指纹数据
        np.savez_compressed(
            output_path,
            **layer_fingerprints
        )
        
        # 保存元数据
        metadata = {
            'model_name': self.model_name,
            'num_triplets': len(knowledge_triplets),
            'target_layers': self.target_layers,
            'triplets': knowledge_triplets[:1000] if len(knowledge_triplets) > 1000 else knowledge_triplets  # 限制元数据大小
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logging.info(f"激活指纹已保存到: {output_path}")
    
    def _load_fingerprints(self, path: str) -> Dict[str, np.ndarray]:
        """
        加载激活指纹
        
        Args:
            path: 文件路径
            
        Returns:
            层级激活指纹
        """
        data = np.load(path)
        return {key: data[key] for key in data.files}
    
    def build_fingerprints_for_text_probes(
        self,
        text_probes: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        为文本探针构建激活指纹
        
        Args:
            text_probes: 文本探针列表
            output_path: 输出路径
            
        Returns:
            激活指纹
        """
        # 将文本转换为伪三元组格式
        pseudo_triplets = []
        for i, text in enumerate(text_probes):
            pseudo_triplets.append({
                'subject': f'probe_{i}',
                'relation': 'text_probe',
                'object': 'unknown',
                'text': text,
                'confidence': 1.0
            })
        
        return self.build_fingerprints(pseudo_triplets, output_path)
    
    def get_layer_dimensions(self) -> Dict[str, int]:
        """
        获取各层的维度信息
        
        Returns:
            层级维度字典
        """
        # 运行一个小样本来获取维度
        sample_text = "This is a test sentence."
        inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
        
        self._register_hooks()
        self.activations.clear()
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        dimensions = {}
        for layer_id, activation in self.activations.items():
            dimensions[layer_id] = activation.shape[-1]  # hidden_dim
        
        self._clear_hooks()
        return dimensions 