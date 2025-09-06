# CausalEditor集成的Llama-3.1模型使用指南

本文档介绍如何使用集成了CausalEditor的Llama-3.1-8B-Instruct模型。

## 主要特性

### 针对Llama-3.1的优化
- **更大的上下文窗口**：支持最大131,072个token的上下文长度
- **改进的RoPE**：使用更大的theta值(500,000)以支持长上下文
- **更大的词汇表**：128,256个token的词汇表
- **优化的内存使用**：针对8B模型规模进行内存优化
- **Flash Attention 2**：默认启用以提升性能
- **bfloat16精度**：更好的数值稳定性

### CausalEditor集成
- **动态冲突检测**：实时检测生成过程中的幻觉
- **激活编辑**：基于RAG检索的知识进行激活层编辑
- **指纹库缓存**：避免重复计算相同片段的激活指纹
- **多模式支持**：支持普通生成和多选题模式

## 快速开始

### 基本使用

```python
from modeling_llama_causal import CausalLlama31ForCausalLM

# 加载模型（自动配置CausalEditor）
model = CausalLlama31ForCausalLM.from_pretrained_with_dynamic_causal_editor(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda",
    rag_config=rag_config  # RAG配置字典
)

# 生成文本
input_text = "What is the capital of France?"
input_ids = model.tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 配置RAG

```python
rag_config = {
    "retrieval": {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "index_path": "/path/to/vector/index"
    },
    "conflict_detector": {
        "threshold": 0.8,
        "token_window": 128,  # 针对8B模型增加窗口大小
        "method": "cosine_similarity"
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
}
```

### 多选题模式

```python
# 设置多选题模式
model.set_generation_mode(is_mc=True, prompt_length=256)

# 或者在生成时指定
outputs = model.generate(
    input_ids,
    is_mc_mode=True,
    prompt_length=256,
    max_new_tokens=50
)
```

## 模型架构差异

### 与Llama-2的主要区别

| 特性 | Llama-2-7B | Llama-3.1-8B |
|------|------------|---------------|
| 参数量 | 7B | 8B |
| 词汇表大小 | 32,000 | 128,256 |
| 最大上下文 | 4,096 | 131,072 |
| RoPE theta | 10,000 | 500,000 |
| 中间层大小 | 11,008 | 14,336 |
| 推荐精度 | float16 | bfloat16 |
| 默认上下文窗口 | 64 | 128 |

## 内存优化建议

### 24GB显存配置
```python
recommended_kwargs = {
    "dtype": torch.bfloat16,
    "device_map": "auto",
    "low_cpu_mem_usage": True,
    "max_memory": {0: "20GB"},  # 为8B模型预留更多内存
    "attn_implementation": "flash_attention_2"
}
```

### 更大显存配置
```python
# 对于40GB+显存
recommended_kwargs = {
    "dtype": torch.bfloat16,
    "device_map": "auto",
    "max_memory": {0: "35GB"},
    "attn_implementation": "flash_attention_2"
}
```

## 性能监控

```python
# 获取模型信息
model_info = model.get_model_info()
print(f"模型类型: {model_info['model_type']}")
print(f"层数: {model_info['num_layers']}")
print(f"隐藏层大小: {model_info['hidden_size']}")
print(f"词汇表大小: {model_info['vocab_size']}")

# 获取CausalEditor统计信息
stats = model.get_causal_editor_statistics()
print(f"编辑次数: {stats.get('edit_count', 0)}")
print(f"冲突检测次数: {stats.get('conflict_count', 0)}")

# 重置统计信息
model.reset_causal_editor_statistics()
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`max_memory`设置
   - 使用`torch.float16`而非`bfloat16`
   - 启用`low_cpu_mem_usage=True`

2. **生成速度慢**
   - 确保安装了Flash Attention 2
   - 检查`attn_implementation="flash_attention_2"`
   - 减少`token_window`大小

3. **CausalEditor导入失败**
   - 检查`causal_editor`模块是否正确安装
   - 确认所有依赖项已安装

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
model = CausalLlama31ForCausalLM.from_pretrained_with_dynamic_causal_editor(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda",
    rag_config=rag_config
)
```

## 与其他版本的兼容性

```python
# 可以同时使用多个版本
from modeling_llama_causal import (
    CausalLlama2ForCausalLM,  # Llama-2版本
    CausalLlama31ForCausalLM  # Llama-3.1版本
)

# 根据需要选择合适的版本
if model_size == "7b":
    model_class = CausalLlama2ForCausalLM
elif model_size == "8b":
    model_class = CausalLlama31ForCausalLM
```

## 注意事项

1. **模型权重兼容性**：确保使用正确的Llama-3.1预训练权重
2. **Tokenizer兼容性**：Llama-3.1使用不同的tokenizer，词汇表更大
3. **上下文长度**：虽然支持长上下文，但会显著增加内存使用
4. **精度选择**：bfloat16在Llama-3.1上表现更好，但需要硬件支持

## 更新日志

- **v1.0**: 初始版本，支持Llama-3.1-8B-Instruct
- 基于Llama-2版本的成熟架构
- 针对8B模型规模优化内存使用
- 支持更长的上下文窗口和改进的RoPE