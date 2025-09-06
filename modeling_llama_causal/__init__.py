"""CausalEditor集成的Llama模型模块

本模块提供了集成CausalEditor的不同版本Llama模型：
- Llama-2版本：针对Llama-2-7b-chat模型优化
- Llama-3.1版本：针对Llama-3.1-8B-Instruct模型优化
"""

try:
    # Llama-2版本
    from .modeling_llama_causal_Llama2 import (
        CausalLlama2ForCausalLM,
        CausalLlama2Model,
        CausalLlama2Attention,
        CausalLlama2MLP,
        CausalLlama2DecoderLayer,
        # 兼容性别名
        CausalLlama2,
        CausalLlamaForCausalLM_Llama2,
    )
except ImportError as e:
    import logging
    logging.warning(f"Failed to import Llama-2 causal models: {e}")
    # 设置为None以便后续检查
    CausalLlama2ForCausalLM = None
    CausalLlama2Model = None
    CausalLlama2Attention = None
    CausalLlama2MLP = None
    CausalLlama2DecoderLayer = None
    CausalLlama2 = None
    CausalLlamaForCausalLM_Llama2 = None

try:
    # Llama-3.1版本
    from .modeling_llama_causal_Llama3_1 import (
        CausalLlama31ForCausalLM,
        CausalLlama31Model,
        CausalLlama31Attention,
        CausalLlama31MLP,
        CausalLlama31DecoderLayer,
        # 兼容性别名
        CausalLlama31,
        CausalLlamaForCausalLM_Llama31,
    )
except ImportError as e:
    import logging
    logging.warning(f"Failed to import Llama-3.1 causal models: {e}")
    # 设置为None以便后续检查
    CausalLlama31ForCausalLM = None
    CausalLlama31Model = None
    CausalLlama31Attention = None
    CausalLlama31MLP = None
    CausalLlama31DecoderLayer = None
    CausalLlama31 = None
    CausalLlamaForCausalLM_Llama31 = None

# 导出所有可用的类
__all__ = [
    # Llama-2版本
    "CausalLlama2ForCausalLM",
    "CausalLlama2Model",
    "CausalLlama2Attention",
    "CausalLlama2MLP",
    "CausalLlama2DecoderLayer",
    "CausalLlama2",
    "CausalLlamaForCausalLM_Llama2",
    # Llama-3.1版本
    "CausalLlama31ForCausalLM",
    "CausalLlama31Model",
    "CausalLlama31Attention",
    "CausalLlama31MLP",
    "CausalLlama31DecoderLayer",
    "CausalLlama31",
    "CausalLlamaForCausalLM_Llama31",
]

# 过滤掉None值（导入失败的类）
__all__ = [name for name in __all__ if globals().get(name) is not None]