"""集成CausalEditor的Llama-3.1模型
基于modeling_llama_causal_Llama2.py，针对Llama-3.1-8B-Instruct模型进行优化
主要改进：
1. 针对Llama-3.1的架构特点进行优化
2. 调整默认参数以适应8B模型规模
3. 增强对Llama-3.1特有特性的支持（如更长的上下文窗口）
4. 优化内存使用和性能，支持更大的模型规模
5. 支持Llama-3.1的新特性如改进的RoPE和注意力机制
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import logging

import torch
import torch.nn.functional as F


# 导入原始的Llama组件（这里假设从transformers导入）
try:
    from transformers.models.llama.modeling_llama import (
        LlamaModel,
        LlamaForCausalLM,
        LlamaAttention,
        LlamaMLP,
        LlamaDecoderLayer,
        LlamaPreTrainedModel,
    )
    from transformers.cache_utils import Cache
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        logging.warning(f'导入失败: {e}')
        # TODO: 添加fallback逻辑

except ImportError as e:
    logging.error(f"Failed to import Llama components: {e}")

    raise ImportError("Required Llama components not available. Please install transformers library.")

try:
    from causal_editor.core.causal_editor import CausalEditor
except ImportError:
    try:
        # 尝试从causal_editor直接导入
        from causal_editor import CausalEditor
    except ImportError:
        # 如果导入失败，稍后会在运行时处理
        CausalEditor = None
        logging.warning("CausalEditor not available. Please check causal_editor module installation.")

# 新增：尝试导入RAG配置加载器
try:
    from causal_editor.dynamic.rag_config import RAGConfig
except Exception as e:
    RAGConfig = None
    logging.debug(f"RAGConfig import failed or not available: {e}")


class CausalLlama31Attention(LlamaAttention):
    """集成CausalEditor的Llama-3.1 Attention层
    针对Llama-3.1的架构特点进行优化
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.causal_editor = None  # 将在模型初始化时设置

        # 保持原始配置以确保与Llama-3.1预训练权重的兼容性
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        
        # Llama-3.1特有的配置
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 131072)  # 3.1支持更长上下文
        self.rope_theta = getattr(config, "rope_theta", 500000.0)  # 3.1使用更大的RoPE theta
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 添加形状日志
        logging.debug(f"🔍 CausalLlama31Attention.forward 层{self.layer_idx} hidden_states.shape: {hidden_states.shape}")
        
        # 原始的attention前向传播
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # 如果设置了CausalEditor，进行激活编辑
        if self.causal_editor is not None and self.layer_idx is not None:
            # 设置当前层ID
            layer_id = f"{self.layer_idx}.attn"
            self.causal_editor.set_current_layer(layer_id)

            # 获取激活状态（attention的输出）
            attention_output = outputs[0]
            logging.debug(f"🔍 CausalLlama31Attention.forward 层{self.layer_idx} attention_output.shape: {attention_output.shape}")

            # 执行CausalEditor编辑
            edited_output = self.causal_editor.edit_activations(
                activations=attention_output,
                generated_tokens=getattr(self, "_current_tokens", None),
                context_tokens=getattr(self, "_context_tokens", None),
                input_text=getattr(self, "_input_text", None),
            )

            # 返回编辑后的结果
            return (edited_output,) + outputs[1:]

        return outputs


class CausalLlama31MLP(LlamaMLP):
    """集成CausalEditor的Llama-3.1 MLP层
    针对Llama-3.1的架构特点进行优化
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.causal_editor = None  # 将在模型初始化时设置
        
        # Llama-3.1特有的MLP配置
        self.intermediate_size = getattr(config, "intermediate_size", 14336)  # 8B模型的中间层大小

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
                generated_tokens=getattr(self, "_current_tokens", None),
                context_tokens=getattr(self, "_context_tokens", None),
                input_text=getattr(self, "_input_text", None),
            )

            return edited_output

        return output


class CausalLlama31DecoderLayer(LlamaDecoderLayer):
    """集成CausalEditor的Llama-3.1 Decoder层
    针对Llama-3.1的架构特点进行优化
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # 替换attention和mlp为我们的Llama-3.1版本
        self.self_attn = CausalLlama31Attention(config, layer_idx)
        self.mlp = CausalLlama31MLP(config, layer_idx)

        self.layer_idx = layer_idx
        self.causal_editor = None

    def set_causal_editor(self, causal_editor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor
        self.self_attn.causal_editor = causal_editor
        self.mlp.causal_editor = causal_editor


class CausalLlama31Model(LlamaModel):
    """集成CausalEditor的Llama-3.1模型
    针对Llama-3.1的架构特点进行优化
    """

    def __init__(self, config):
        # 保持原始的预训练配置以确保与Llama-3.1权重的兼容性
        super().__init__(config)

        # 替换layers为我们的Llama-3.1版本
        self.layers = torch.nn.ModuleList(
            [
                CausalLlama31DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.causal_editor = None
        self.current_tokens = []
        self.context_tokens = []
        
        # Llama-3.1特有的配置
        self.vocab_size = getattr(config, "vocab_size", 128256)  # 3.1使用更大的词汇表
        self.hidden_size = getattr(config, "hidden_size", 4096)  # 8B模型的隐藏层大小

    def set_causal_editor(self, causal_editor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor

        # 为所有层设置CausalEditor
        for layer in self.layers:
            layer.set_causal_editor(causal_editor)

    def set_generation_context(
        self, generated_tokens: List[str], context_tokens: List[str], input_text: Optional[str] = None
    ):
        """设置生成上下文信息"""
        self.current_tokens = generated_tokens
        self.context_tokens = context_tokens

        # 传播到所有层
        for layer in self.layers:
            layer.self_attn._current_tokens = generated_tokens
            layer.self_attn._context_tokens = context_tokens
            layer.self_attn._input_text = input_text
            layer.mlp._current_tokens = generated_tokens
            layer.mlp._context_tokens = context_tokens
            layer.mlp._input_text = input_text


class CausalLlama31ForCausalLM(LlamaForCausalLM):
    """集成CausalEditor的Llama-3.1因果语言模型
    针对Llama-3.1-8B-Instruct模型进行优化
    """

    def __init__(self, config):
        super().__init__(config)

        # 替换模型为我们的Llama-3.1版本
        self.model = CausalLlama31Model(config)

        self.causal_editor = None
        self.tokenizer = None  # 需要外部设置
        self.mc_mode = False # 是否为多选题处理逻辑
        self.prompt_length = None
        self._default_generation_kwargs: Dict[str, Any] = {}

    @classmethod
    def from_pretrained_with_dynamic_causal_editor(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        rag_config: dict = None,
        **kwargs,
    ):
        # 针对Llama-3.1-8B模型的推荐配置
        recommended_kwargs = {
            "dtype": torch.bfloat16,  # 3.1推荐使用bfloat16以获得更好的数值稳定性
            "device_map": "auto",  # 自动分发到可用设备
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
            "trust_remote_code": True,  # 信任远程代码
            "max_memory": {0: "20GB"},  # 针对24GB显存优化，为8B模型预留更多内存
            "attn_implementation": "flash_attention_2",  # 使用Flash Attention 2优化
        }
        
        # 合并用户提供的kwargs，用户参数优先
        final_kwargs = {**recommended_kwargs, **kwargs}
        
        # 加载基础Llama-3.1模型
        model = cls.from_pretrained(model_name_or_path, **final_kwargs)
        
        # 加载tokenizer用于动态模式
        tokenizer_kwargs = {'trust_remote_code': True}
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 检查CausalEditor是否可用
        if CausalEditor is None:
            raise ImportError("CausalEditor无法导入，请检查causal_editor模块是否正确安装")
        
        # 针对Llama-3.1-8B模型初始化动态CausalEditor
        causal_editor = CausalEditor(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            device=device,
            rag_config=rag_config,
        )

        # 设置到模型中
        model.set_causal_editor(causal_editor)
        model.set_tokenizer(tokenizer)

        # 从配置中提取生成默认参数
        try:
            model._default_generation_kwargs = model._extract_generation_defaults_from_config(
                getattr(causal_editor, 'rag_config', None)
            )
            if model._default_generation_kwargs:
                logging.info(f"默认生成参数（来自配置）: {model._default_generation_kwargs}")
        except Exception as e:
            logging.debug(f"提取默认生成参数失败: {e}")

        logging.info(f"CausalLlama31模型已加载并配置动态CausalEditor")
        logging.info(f"模型: {model_name_or_path}")
        
        return model

    def set_causal_editor(self, causal_editor):
        """设置CausalEditor实例"""
        self.causal_editor = causal_editor
        self.model.set_causal_editor(causal_editor)

    def set_tokenizer(self, tokenizer):
        """设置tokenizer用于token解析"""
        self.tokenizer = tokenizer

    def set_generation_mode(
        self, is_mc: bool = False, prompt_length: Optional[int] = None
    ):
        """设置生成模式"""
        self.mc_mode = is_mc
        self.prompt_length = prompt_length

        if self.causal_editor:
            self.causal_editor.set_generation_mode(is_mc, prompt_length)

    def _extract_generation_defaults_from_config(self, rag_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """从RAG配置中提取生成参数默认值（支持 generation 或 evaluation 段）。"""
        defaults: Dict[str, Any] = {}
        if not rag_cfg or not isinstance(rag_cfg, dict):
            return defaults
        try:
            # 兼容两个命名
            gen_cfg = rag_cfg.get('generation') or rag_cfg.get('evaluation') or {}
            if not isinstance(gen_cfg, dict):
                return defaults
            allowed_keys = {
                'max_new_tokens', 'do_sample', 'temperature', 'top_k', 'top_p',
                'num_beams', 'repetition_penalty', 'eos_token_id', 'pad_token_id',
                # 模式相关
                'is_mc_mode', 'prompt_length'
            }
            for k, v in gen_cfg.items():
                if k in allowed_keys:
                    defaults[k] = v
            # 简单的"mode"语义映射（可选）
            mode = gen_cfg.get('mode')
            if mode == 'deterministic':
                defaults.setdefault('do_sample', False)
                defaults.setdefault('temperature', 1.0)
            elif mode == 'creative':
                defaults.setdefault('do_sample', True)
                defaults.setdefault('temperature', 0.9)
            elif mode == 'mc_qa':
                defaults.setdefault('is_mc_mode', True)
                # prompt_length可由配置显式提供
        except Exception as e:
            logging.debug(f"解析生成配置失败: {e}")
        return defaults
    
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
        **kwargs,
    ):
        # 添加形状日志以定位batch扩展来源
        if input_ids is not None:
            logging.debug(f"🔍 CausalLlama31ForCausalLM.forward 入口 input_ids.shape: {input_ids.shape}")
        if attention_mask is not None:
            logging.debug(f"🔍 CausalLlama31ForCausalLM.forward 入口 attention_mask.shape: {attention_mask.shape}")
        try:
            # 检查CausalEditor和动态模式是否可用
            if (hasattr(self, 'causal_editor') and self.causal_editor and 
                hasattr(self, 'tokenizer') and self.tokenizer and input_ids is not None):
                try:
                    # 确保input_ids是有效的tensor
                    if input_ids.numel() > 0:
                        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                        user_input_text = self.causal_editor.get_current_input()

                        if user_input_text is not None and hasattr(self.model, 'set_generation_context'):
                            # 读取上下文窗口：默认128（为8B模型增加），可由 rag_config['conflict_detector']['token_window'] 覆盖
                            context_window = 128
                            try:
                                if hasattr(self, 'causal_editor') and self.causal_editor:
                                    rag_cfg = getattr(self.causal_editor, 'rag_config', {}) or {}
                                    conflict_cfg = rag_cfg.get('conflict_detector', {}) or {}
                                    cfg_window = conflict_cfg.get('token_window', None)
                                    if cfg_window is not None:
                                        try:
                                            cfg_window_int = int(cfg_window)
                                            if cfg_window_int > 0:
                                                context_window = cfg_window_int
                                        except Exception:
                                            logging.debug(f"Invalid token_window in rag_config: {cfg_window}, using default 128")
                            except Exception as cfg_e:
                                logging.debug(f"Falling back to default context_window=128 due to config error: {cfg_e}")

                            self.model.set_generation_context(
                                generated_tokens=tokens[-context_window:] if len(tokens) > context_window else tokens,
                                context_tokens=tokens[:-context_window] if len(tokens) > context_window else [],
                                input_text=user_input_text,
                            )
                            logging.debug(f"Set generation context with window={context_window}, total_tokens={len(tokens)}")
                except Exception as e:
                    logging.warning(f"Token parsing or context setting failed: {e}, proceeding without context.")

            result = super().forward(
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
                **kwargs,
            )
            return result
        except Exception as e:
            logging.error(f"Error during forward propagation: {e}")
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
                **kwargs,
            )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus: Optional[bool] = None,
        assistant_model=None,
        streamer=None,
        **kwargs,
    ):
        """重写generate方法以支持CausalEditor
        针对Llama-3.1模型进行了优化
        """
        try:
            # 先从配置注入默认生成参数（用户传参优先）
            defaults = getattr(self, '_default_generation_kwargs', None)
            if defaults and generation_config is None:
                for k, v in defaults.items():
                    if k in ('is_mc_mode', 'prompt_length'):
                        continue  # 模式单独处理
                    if k not in kwargs:
                        kwargs[k] = v

            if hasattr(self, 'causal_editor') and self.causal_editor:
                # 读取是否为MC模式与提示长度（优先级：kwargs > defaults > 属性）
                is_mc = kwargs.get("is_mc_mode", defaults.get('is_mc_mode') if isinstance(defaults, dict) else None)
                if is_mc is None:
                    is_mc = getattr(self, 'mc_mode', False)
                prompt_length = kwargs.get("prompt_length", defaults.get('prompt_length') if isinstance(defaults, dict) else None)
                if prompt_length is None:
                    prompt_length = getattr(self, 'prompt_length', None)

                if hasattr(self.causal_editor, 'set_generation_mode'):
                    self.causal_editor.set_generation_mode(bool(is_mc), prompt_length)
                    logging.debug(f"Set generation mode: is_mc={bool(is_mc)}, prompt_length={prompt_length}")

                # 避免向上游forward传播自定义控制参数
                kwargs.pop('is_mc_mode', None)
                kwargs.pop('prompt_length', None)
        except Exception as e:
            logging.warning(f"Failed to set generation mode or inject defaults: {e}, proceeding with default settings")

        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )

    def get_causal_editor_statistics(self) -> Dict[str, Any]:
        """获取CausalEditor统计信息"""
        try:
            if hasattr(self, 'causal_editor') and self.causal_editor and hasattr(self.causal_editor, 'get_statistics'):
                return self.causal_editor.get_statistics()
        except Exception as e:
            logging.error(f"Failed to get CausalEditor statistics: {e}")
        return {}

    def reset_causal_editor_statistics(self):
        """重置CausalEditor统计信息"""
        try:
            if hasattr(self, 'causal_editor') and self.causal_editor and hasattr(self.causal_editor, 'reset_statistics'):
                self.causal_editor.reset_statistics()
                logging.debug("CausalEditor statistics reset successfully")
        except Exception as e:
            logging.error(f"Failed to reset CausalEditor statistics: {e}")

    def save_causal_editor_config(self, path: str):
        """保存CausalEditor配置"""
        try:
            if hasattr(self, 'causal_editor') and self.causal_editor and hasattr(self.causal_editor, 'save_config'):
                self.causal_editor.save_config(path)
                logging.info(f"CausalEditor config saved to {path}")
            else:
                logging.warning("CausalEditor not available or save_config method not found")
        except Exception as e:
            logging.error(f"Failed to save CausalEditor config: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息，用于调试和监控"""
        try:
            info = {
                "model_type": "CausalLlama31ForCausalLM",
                "num_layers": len(self.model.layers) if hasattr(self.model, 'layers') else "unknown",
                "hidden_size": getattr(self.config, "hidden_size", None),
                "vocab_size": getattr(self.config, "vocab_size", None),
                "num_attention_heads": getattr(self.config, "num_attention_heads", None),
                "intermediate_size": getattr(self.config, "intermediate_size", None),
                "max_position_embeddings": getattr(self.config, "max_position_embeddings", None),
                "rope_theta": getattr(self.config, "rope_theta", None),
                "causal_editor_attached": hasattr(self, 'causal_editor') and self.causal_editor is not None,
                "tokenizer_attached": hasattr(self, 'tokenizer') and self.tokenizer is not None,
                "device": str(next(self.parameters()).device) if list(self.parameters()) else "unknown",
            }
                    
            return info
        except Exception as e:
            logging.error(f"Failed to get model info: {e}")
            return {"error": str(e)}


# 兼容性别名
CausalLlama31 = CausalLlama31ForCausalLM

# 为了向后兼容，也提供原始名称的别名
CausalLlamaForCausalLM_Llama31 = CausalLlama31ForCausalLM