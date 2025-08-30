# CausalEditor API 接口文档

## 概述

CausalEditor 是一个用于大语言模型因果编辑的Python框架，支持动态模式下的实时冲突检测和激活状态编辑。本文档详细介绍了框架中每个组件的API接口。

## 目录

1. [核心组件](#核心组件)
   - [CausalEditor](#causaleditormain)
   - [CausalConflictDetector](#causalconflictdetector)
   - [CounterfactualEditor](#counterfactualeditor)
2. [动态模块](#动态模块)
   - [DynamicCandidateFilter](#dynamiccandidatefilter)
   - [DynamicFingerprintBuilder](#dynamicfingerprintbuilder)
   - [DynamicVectorIndex](#dynamicvectorindex)
3. [工具模块](#工具模块)
   - [EntityExtractionManager](#entityextractionmanager)
   - [WebKnowledgeRetriever](#webknowledgeretriever)

---

## 核心组件

### CausalEditor(main)

主要的因果编辑器类，协调所有组件进行因果冲突检测和编辑。

#### 初始化

```python
CausalEditor(
    model,
    tokenizer,
    target_layers=None,
    similarity_threshold=0.85,
    conflict_threshold=0.7,
    edit_strength=1.0,
    min_confidence=0.6,
    use_dynamic_mode=True,
    device='auto',
    debug_mode=False
)
```

**参数说明：**
- `model`: 要编辑的语言模型
- `tokenizer`: 对应的分词器
- `target_layers`: 目标编辑层列表，默认为模型的中间层
- `similarity_threshold`: 相似度阈值，用于指纹匹配
- `conflict_threshold`: 冲突判定阈值
- `edit_strength`: 编辑强度
- `min_confidence`: 最小置信度阈值
- `use_dynamic_mode`: 是否启用动态模式
- `device`: 计算设备
- `debug_mode`: 是否启用调试模式

#### 主要方法

##### `prepare_for_input(input_text: str)`

为输入文本准备动态候选知识。

**参数：**
- `input_text`: 输入文本

**功能：**
- 提取实体和关系
- 筛选相关候选三元组
- 构建激活指纹
- 更新向量索引

##### `detect_and_edit(hidden_states, layer_idx, generated_tokens, context_tokens, input_text)`

检测冲突并执行编辑的核心方法。

**参数：**
- `hidden_states`: 当前层的隐藏状态
- `layer_idx`: 当前层索引
- `generated_tokens`: 已生成的token列表
- `context_tokens`: 上下文token列表
- `input_text`: 原始输入文本

**返回：**
- 编辑后的隐藏状态（如果发生编辑）或原始隐藏状态

##### `get_statistics()`

获取编辑器的统计信息。

**返回：**
```python
{
    'conflict_detector_stats': {...},
    'counterfactual_editor_stats': {...},
    'dynamic_candidate_filter_stats': {...},
    'dynamic_fingerprint_builder_stats': {...},
    'dynamic_vector_index_stats': {...},
    'use_dynamic_mode': bool,
    'dynamic_index_size': int
}
```

##### `reset_statistics()`

重置所有组件的统计信息。

##### `update_config(**kwargs)`

更新配置参数。

**支持的参数：**
- `similarity_threshold`
- `conflict_threshold`
- `edit_strength`
- `min_confidence`

---

### CausalConflictDetector

因果冲突检测器，通过实时激活监测和动态指纹比对检测因果断裂点。

#### 初始化

```python
CausalConflictDetector(
    similarity_threshold=0.85,
    conflict_threshold=0.7,
    entity_patterns=None
)
```

**参数说明：**
- `similarity_threshold`: 相似度阈值
- `conflict_threshold`: 冲突判定阈值
- `entity_patterns`: 实体识别正则表达式模式

#### 主要方法

##### `detect_conflict(hidden_states, layer_idx, generated_tokens, context_tokens, input_text, vector_index)`

动态模式下的因果冲突检测。

**参数：**
- `hidden_states`: 当前隐藏状态
- `layer_idx`: 层索引
- `generated_tokens`: 已生成token
- `context_tokens`: 上下文token
- `input_text`: 输入文本
- `vector_index`: 向量索引对象

**返回：**
```python
{
    'has_conflict': bool,
    'conflict_score': float,
    'matched_candidates': list,
    'correction_strength': float
}
```

##### `analyze_dynamic_conflict(matched_candidates, generated_tokens, context_tokens)`

分析动态冲突并计算修正强度。

**参数：**
- `matched_candidates`: 匹配的候选列表
- `generated_tokens`: 已生成token
- `context_tokens`: 上下文token

**返回：**
- 修正强度值（0.0-1.0）

##### `get_conflict_patterns()`

获取冲突模式分析。

**返回：**
```python
{
    'entity_conflicts': int,
    'relation_conflicts': int,
    'temporal_conflicts': int,
    'factual_conflicts': int
}
```

##### `update_thresholds(similarity_threshold=None, conflict_threshold=None)`

更新阈值参数。

---

### CounterfactualEditor

反事实编辑器，基于检测到的因果冲突执行精确的激活状态编辑。

#### 初始化

```python
CounterfactualEditor(
    edit_strength=1.0,
    min_confidence=0.6,
    device='cuda',
    hidden_dim=None
)
```

#### 主要方法

##### `edit(hidden_states, conflict_info, layer_idx, mode='generation')`

执行反事实编辑。

**参数：**
- `hidden_states`: 当前隐藏状态
- `conflict_info`: 冲突信息字典
- `layer_idx`: 层索引
- `mode`: 编辑模式（'MC'或'generation'）

**返回：**
- 编辑后的隐藏状态

##### `get_edit_magnitude_stats()`

获取编辑幅度统计。

**返回：**
```python
{
    'mean_magnitude': float,
    'std_magnitude': float,
    'max_magnitude': float,
    'min_magnitude': float
}
```

##### `update_edit_strength(new_strength: float)`

更新编辑强度。

---

## 动态模块

### DynamicCandidateFilter

动态候选过滤器，基于输入文本实时筛选相关的候选知识三元组。

#### 初始化

```python
DynamicCandidateFilter(
    tokenizer,
    entity_extraction_manager,
    knowledge_base_path=None,
    enable_web_retrieval=True,
    cache_size=1000
)
```

#### 主要方法

##### `extract_entities(text: str)`

从文本中提取实体。

**参数：**
- `text`: 输入文本

**返回：**
```python
[
    {
        'text': str,
        'start': int,
        'end': int,
        'confidence': float,
        'type': str
    }
]
```

##### `infer_relations(text: str, entities: list)`

从文本和实体中推断可能的关系。

**参数：**
- `text`: 输入文本
- `entities`: 实体列表

**返回：**
- 推断的关系列表

##### `filter_candidates(text: str, max_candidates=50)`

筛选候选三元组。

**参数：**
- `text`: 输入文本
- `max_candidates`: 最大候选数量

**返回：**
```python
[
    {
        'subject': str,
        'relation': str,
        'object': str,
        'confidence': float,
        'source': str,
        'match_score': float
    }
]
```

##### `set_web_retrieval_strategy(strategy: str)`

设置网络获取策略。

**支持的策略：**
- 'conservative': 保守策略
- 'balanced': 平衡策略
- 'aggressive': 激进策略

##### `toggle_web_retrieval(enabled: bool)`

启用或禁用网络知识获取。

---

### DynamicFingerprintBuilder

动态指纹构建器，实时为候选三元组生成LLM各层的激活指纹。

#### 初始化

```python
DynamicFingerprintBuilder(
    model,
    tokenizer,
    target_layers,
    device='cuda',
    max_batch_size=8,
    cache_size=500,
    fingerprint_dim=None
)
```

#### 主要方法

##### `build_fingerprints(candidates: list)`

为候选三元组构建激活指纹。

**参数：**
- `candidates`: 候选三元组列表

**返回：**
```python
{
    layer_idx: {
        'fingerprints': torch.Tensor,
        'metadata': list
    }
}
```

##### `clear_cache()`

清空指纹缓存。

---

### DynamicVectorIndex

动态向量索引，实时构建、增量更新和高效查询向量索引。

#### 初始化

```python
DynamicVectorIndex(
    dimension,
    device='cuda',
    index_type='flat',
    max_vectors=10000,
    similarity_threshold=0.8
)
```

**支持的索引类型：**
- 'flat': 平面索引
- 'hnsw': 分层导航小世界图
- 'ivf': 倒排文件索引

#### 主要方法

##### `add_vectors(layer_idx: int, vectors: torch.Tensor, metadata: list)`

添加向量和元数据到索引。

**参数：**
- `layer_idx`: 层索引
- `vectors`: 向量张量
- `metadata`: 元数据列表

##### `search(layer_idx: int, query_vector: torch.Tensor, k=10, score_threshold=None)`

搜索相似向量。

**参数：**
- `layer_idx`: 层索引
- `query_vector`: 查询向量
- `k`: 返回的最近邻数量
- `score_threshold`: 分数阈值

**返回：**
```python
{
    'scores': list,
    'metadata': list,
    'indices': list
}
```

##### `search_all_layers(query_vectors: dict, k=10)`

在所有层中搜索。

##### `get_layer_info(layer_idx: int)`

获取层信息。

**返回：**
```python
{
    'vector_count': int,
    'dimension': int,
    'index_type': str
}
```

##### `clear_layer(layer_idx: int)`

清空指定层的索引。

##### `clear_all()`

清空所有索引。

---

## 工具模块

### EntityExtractionManager

实体提取管理器，支持多种实体提取方法。

#### 主要方法

##### `extract_entities(text: str, method='regex')`

提取实体。

**支持的方法：**
- 'regex': 正则表达式
- 'spacy': SpaCy NER
- 'transformers': Transformers NER

##### `set_extraction_method(method: str)`

设置提取方法。

##### `get_available_methods()`

获取可用的提取方法。

---

### WebKnowledgeRetriever

网络知识获取器，从网络获取相关知识三元组。

#### 主要方法

##### `retrieve_knowledge(entities: list, relations: list, max_results=20)`

获取知识三元组。

**参数：**
- `entities`: 实体列表
- `relations`: 关系列表
- `max_results`: 最大结果数量

**返回：**
- 知识三元组列表

##### `set_retrieval_strategy(strategy: str)`

设置获取策略。

---

## 使用示例

### 基本使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_editor import CausalEditor

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")

# 创建因果编辑器
editor = CausalEditor(
    model=model,
    tokenizer=tokenizer,
    use_dynamic_mode=True,
    debug_mode=True
)

# 准备输入
input_text = "What is the capital of France?"
editor.prepare_for_input(input_text)

# 生成文本（自动进行冲突检测和编辑）
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

# 获取统计信息
stats = editor.get_statistics()
print(f"检测到 {stats['conflict_detector_stats']['conflict_count']} 个冲突")
print(f"执行了 {stats['counterfactual_editor_stats']['edit_count']} 次编辑")
```

### 高级配置

```python
# 自定义配置
editor = CausalEditor(
    model=model,
    tokenizer=tokenizer,
    target_layers=[10, 15, 20],  # 指定编辑层
    similarity_threshold=0.9,    # 提高相似度阈值
    conflict_threshold=0.8,      # 提高冲突阈值
    edit_strength=0.8,           # 降低编辑强度
    use_dynamic_mode=True
)

# 动态更新配置
editor.update_config(
    similarity_threshold=0.85,
    edit_strength=1.2
)

# 设置网络获取策略
editor.dynamic_candidate_filter.set_web_retrieval_strategy('conservative')

# 设置实体提取方法
editor.dynamic_candidate_filter.set_entity_extraction_method('spacy')
```

---

## 错误处理

框架提供了完善的错误处理机制：

- **模型兼容性检查**：自动检测模型架构兼容性
- **设备管理**：自动处理GPU/CPU设备切换
- **内存管理**：自动清理缓存防止内存溢出
- **异常恢复**：编辑失败时自动回退到原始状态

## 性能优化建议

1. **批处理**：使用适当的批处理大小以平衡速度和内存使用
2. **缓存**：启用缓存以避免重复计算
3. **层选择**：选择关键层进行编辑以减少计算开销
4. **阈值调优**：根据具体任务调整相似度和冲突阈值
5. **索引类型**：根据数据规模选择合适的向量索引类型

## 调试和监控

- 启用 `debug_mode` 获取详细的执行信息
- 使用 `get_statistics()` 监控性能指标
- 定期调用 `reset_statistics()` 重置统计信息
- 使用日志记录跟踪编辑过程

---

*本文档基于 CausalEditor v1.0 编写，如有更新请参考最新版本。*