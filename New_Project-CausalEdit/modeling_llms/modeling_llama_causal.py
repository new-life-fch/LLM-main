"""
集成CausalEditor的Llama模型
基于原有的modeling_llama.py，将TruthX替换为CausalEditor
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Dict, Any
import logging

# 导入原始的Llama组件（这里假设从transformers导入）
try:
    from transformers.models.llama.modeling_llama import (
        LlamaModel, LlamaForCausalLM, LlamaAttention, LlamaMLP,
        LlamaDecoderLayer, LlamaPreTrainedModel
    )
except ImportError:
    # 如果在本地环境，可能需要从TruthX的modeling_llama导入
    pass

from causal_editor import CausalEditor


class CausalLlamaAttention(LlamaAttention):
    """
    集成CausalEditor的Llama Attention层
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.causal_editor = None  # 将在模型初始化时设置
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # 原始的attention前向传播
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        # 如果设置了CausalEditor，进行激活编辑
        if self.causal_editor is not None and self.layer_idx is not None:
            # 设置当前层ID
            layer_id = f"{self.layer_idx}.attn"
            self.causal_editor.set_current_layer(layer_id)
            
            # 获取激活状态（attention的输出）
            attention_output = outputs[0]
            
            # 执行CausalEditor编辑
            edited_output = self.causal_editor.edit_activations(
                activations=attention_output,
                generated_tokens=getattr(self, '_current_tokens', None),
                context_tokens=getattr(self, '_context_tokens', None)
            )
            
            # 返回编辑后的结果
            return (edited_output,) + outputs[1:]
        
        return outputs


class CausalLlamaMLP(LlamaMLP):
    """
    集成CausalEditor的Llama MLP层
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.causal_editor = None  # 将在模型初始化时设置
    
    def forward(self, x):
        # 原始的MLP前向传播
        output = super().forward(x)
        
        # 如果设置了CausalEditor，进行激活编辑
        if self.causal_editor is not None and self.layer_idx is not None:
            # 设置当前层ID
            layer_id = f"{self.layer_idx}.mlp"
            self.causal_editor.set_current_layer(layer_id)
            
            # 执行CausalEditor编辑
            edited_output = self.causal_editor.edit_activations(
                activations=output,
                generated_tokens=getattr(self, '_current_tokens', None),
                context_tokens=getattr(self, '_context_tokens', None)
            )
            
            return edited_output
        
        return output


class CausalLlamaDecoderLayer(LlamaDecoderLayer):
    """
    集成CausalEditor的Llama Decoder层
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # 替换attention和mlp为我们的版本
        self.self_attn = CausalLlamaAttention(config, layer_idx)
        self.mlp = CausalLlamaMLP(config, layer_idx)
        
        self.layer_idx = layer_idx
        self.causal_editor = None
    
    def set_causal_editor(self, causal_editor: CausalEditor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor
        self.self_attn.causal_editor = causal_editor
        self.mlp.causal_editor = causal_editor


class CausalLlamaModel(LlamaModel):
    """
    集成CausalEditor的Llama模型
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 替换layers为我们的版本
        self.layers = torch.nn.ModuleList([
            CausalLlamaDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.causal_editor = None
        self.current_tokens = []
        self.context_tokens = []
    
    def set_causal_editor(self, causal_editor: CausalEditor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor
        
        # 为所有层设置CausalEditor
        for layer in self.layers:
            layer.set_causal_editor(causal_editor)
        
        logging.info("CausalEditor已设置到模型中")
    
    def set_generation_context(self, generated_tokens: List[str], context_tokens: List[str]):
        """设置生成上下文信息"""
        self.current_tokens = generated_tokens
        self.context_tokens = context_tokens
        
        # 传播到所有层
        for layer in self.layers:
            layer.self_attn._current_tokens = generated_tokens
            layer.self_attn._context_tokens = context_tokens
            layer.mlp._current_tokens = generated_tokens
            layer.mlp._context_tokens = context_tokens


class CausalLlamaForCausalLM(LlamaForCausalLM):
    """
    集成CausalEditor的Llama因果语言模型
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 替换模型为我们的版本
        self.model = CausalLlamaModel(config)
        
        self.causal_editor = None
        self.tokenizer = None  # 需要外部设置
        self.mc_mode = False
        self.prompt_length = None
    
    @classmethod
    def from_pretrained_with_causal_editor(
        cls,
        model_name_or_path: str,
        vector_db_path: str,
        edit_strength: float = 1.0,
        top_layers: int = 10,
        similarity_threshold: float = 0.8,
        conflict_threshold: float = 0.6,
        device: str = "cuda",
        **kwargs
    ):
        """
        从预训练模型加载并初始化CausalEditor
        
        Args:
            model_name_or_path: 预训练模型路径
            vector_db_path: 预计算的向量数据库路径
            edit_strength: 编辑强度
            top_layers: 参与编辑的层数
            similarity_threshold: 相似度阈值
            conflict_threshold: 冲突检测阈值
            device: 设备
            **kwargs: 其他参数
            
        Returns:
            初始化好的模型实例
        """
        # 加载基础模型
        model = cls.from_pretrained(model_name_or_path, **kwargs)
        
        # 初始化CausalEditor
        causal_editor = CausalEditor(
            vector_db_path=vector_db_path,
            model_name=model_name_or_path,
            edit_strength=edit_strength,
            top_layers=top_layers,
            similarity_threshold=similarity_threshold,
            conflict_threshold=conflict_threshold,
            device=device
        )
        
        # 设置到模型中
        model.set_causal_editor(causal_editor)
        
        logging.info(f"CausalLlama模型已从 {model_name_or_path} 加载并配置CausalEditor")
        return model
    
    def set_causal_editor(self, causal_editor: CausalEditor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor
        self.model.set_causal_editor(causal_editor)
    
    def set_tokenizer(self, tokenizer):
        """设置tokenizer用于token解析"""
        self.tokenizer = tokenizer
    
    def set_generation_mode(self, is_mc: bool = False, prompt_length: Optional[int] = None):
        """设置生成模式"""
        self.mc_mode = is_mc
        self.prompt_length = prompt_length
        
        if self.causal_editor:
            self.causal_editor.set_generation_mode(is_mc, prompt_length)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # 如果有tokenizer，解析当前tokens
        if self.tokenizer is not None and input_ids is not None:
            try:
                # 解析token为文本
                current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # 更新生成上下文
                self.model.set_generation_context(
                    generated_tokens=tokens[-10:] if len(tokens) > 10 else tokens,  # 最近的tokens
                    context_tokens=tokens[:-10] if len(tokens) > 10 else []
                )
            except Exception as e:
                logging.debug(f"Token解析失败: {e}")
        
        # 调用原始forward
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus: Optional[bool] = None,
        assistant_model = None,
        streamer = None,
        **kwargs,
    ):
        """
        覆盖generate方法以支持CausalEditor
        """
        # 设置生成模式
        if self.causal_editor:
            is_mc = kwargs.get('is_mc_mode', self.mc_mode)
            prompt_length = kwargs.get('prompt_length', self.prompt_length)
            self.causal_editor.set_generation_mode(is_mc, prompt_length)
        
        # 调用原始generate
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs
        )
    
    def get_causal_editor_statistics(self) -> Dict[str, Any]:
        """获取CausalEditor统计信息"""
        if self.causal_editor:
            return self.causal_editor.get_statistics()
        return {}
    
    def reset_causal_editor_statistics(self):
        """重置CausalEditor统计信息"""
        if self.causal_editor:
            self.causal_editor.reset_statistics()
    
    def save_causal_editor_config(self, path: str):
        """保存CausalEditor配置"""
        if self.causal_editor:
            self.causal_editor.save_config(path)


# 兼容性别名
CausalLlama = CausalLlamaForCausalLM 