# CausalEditor Pipeline 使用指南

这个自定义Pipeline将原始的`truthfulqa_llama2_causal_editor.py`脚本中的模型加载和生成流程集成到FlashRAG框架中，实现了RAG检索与因果编辑的结合。支持FlashRAG框架的各种数据集，如Natural Questions (NQ)、MS MARCO、HotpotQA等。

## 项目总体逻辑

1. **构建RAG语料库**（开发测试阶段选择采样少量文章）（一次性工作）
2. **按照用户问题检索top-k个片段**
3. **检索到的片段经过处理用LLM进行前向传播构建激活指纹库**，每个片段对应一组指纹，如果将来检索到同样片段，不用再次进行前向传播
4. **开始推理，推理时通过动态阈值判断是否出现幻觉**，即冲突检测
5. **如果检测到冲突，则利用激活指纹库的片段进行激活编辑**

## 文件结构

```
scripts/
├── causal_editor_pipeline.py          # 自定义CausalEditor Pipeline实现
├── run_causal_editor_pipeline.py      # 使用示例脚本
├── README_CausalEditor_Pipeline.md    # 本文档
└── truthfulqa_llama2_causal_editor.py # 原始脚本（参考）
```

## 核心组件

### 1. CausalEditorPipeline 类

继承自FlashRAG的`BasicPipeline`，集成了以下功能：

- **模型加载**: 使用`CausalLlama2ForCausalLM.from_pretrained_with_dynamic_causal_editor`加载带因果编辑功能的Llama-2模型
- **RAG检索**: 基于FlashRAG框架的检索功能
- **激活指纹库**: 构建和管理检索片段的激活指纹
- **冲突检测**: 动态阈值判断是否出现幻觉
- **激活编辑**: 利用激活指纹库进行因果编辑

### 2. 主要方法

- `load_model()`: 加载CausalEditor模型
- `initialize_retriever()`: 初始化RAG检索器
- `prepare_input()`: 准备模型输入
- `generate_answer()`: 生成答案（包含冲突检测和激活编辑）
- `run()`: 运行完整的pipeline
- `save_results()`: 保存结果到文件

## 使用方法

### 1. 环境准备

1. **安装依赖**：
   ```bash
   pip install flashrag torch transformers pyyaml
   ```

2. **准备配置文件**：
   - `configs/retrieval_config.yaml`: 检索器和pipeline配置
   - `configs/causal_editor.json`: CausalEditor框架配置

3. **准备模型文件**：
   ```bash
   # 模型文件
   /root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/llama2-7b-chat-hf/
   ```

4. **数据集**：
   - FlashRAG会自动下载和管理数据集

### 2. 直接运行示例

```bash
cd /root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit
python scripts/run_causal_editor_pipeline.py
```

### 3. 自定义使用

```python
from scripts.causal_editor_pipeline import CausalEditorPipeline
from flashrag import Config
import yaml

# 配置文件路径
retrieval_config_path = "./configs/retrieval_config.yaml"
causal_editor_config_path = "./configs/causal_editor.json"

# 加载检索配置
with open(retrieval_config_path, 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)

# 覆盖测试参数
config_dict.update({
    'dataset_name': 'nq',
    'test_sample_num': 10,
    'do_eval': True,
    'metrics': ['em', 'f1', 'sub_em']
})

config = Config(config_dict=config_dict)

# 初始化Pipeline
pipeline = CausalEditorPipeline(
    config=config,
    model_name="/path/to/llama2-7b-chat-hf",
    causal_editor_config_path=causal_editor_config_path,
    max_length=4096,
    max_new_tokens=50
)

# 运行Pipeline
results = pipeline.run()
print(f"处理完成，结果已保存")
```

## 配置参数说明

### Pipeline配置

- `config`: FlashRAG配置对象 (从retrieval_config.yaml创建)
- `model_name`: Llama-2模型路径
- `causal_editor_config_path`: CausalEditor配置文件路径
- `max_length`: 最大输入长度
- `max_new_tokens`: 最大生成token数
- `device`: 计算设备（"cuda" 或 "cpu"）

### 检索器配置 (retrieval_config.yaml)

主要配置参数：

- `data_dir`: 数据集目录
- `save_dir`: 结果保存目录
- `dataset_name`: 数据集名称 (如 'nq', 'msmarco')
- `split`: 数据集分割 (['train', 'dev', 'test'])
- `test_sample_num`: 测试样本数量
- `retrieval_method`: 检索方法 (如 'e5', 'bm25')
- `corpus_path`: 语料库路径
- `index_path`: 索引文件路径
- `model2path`: 模型路径映射
- `model2pooling`: 模型池化方法

### CausalEditor配置 (causal_editor.json)

主要配置参数：

- `model`: 模型配置 (name, device, batch_size, dtype)
- `fingerprint_builder`: 指纹构建器配置
- `vector_index`: 向量索引配置
- `conflict_detector`: 冲突检测器配置
- `counterfactual_editor`: 反事实编辑器配置
- `generation`: 生成参数配置

## 输出文件

Pipeline运行完成后会生成以下文件：

```
result/result_causal_editor_pipeline/
├── predictions.json          # 预测结果（JSON格式）
├── predictions.jsonl         # 预测结果（JSONL格式）
├── detailed_results.json     # 详细结果（包含成功/失败状态）
└── statistics.json           # 统计信息
```

### 结果格式

每个结果项包含：

```json
{
    "question": "问题文本",
    "golden_answers": ["标准答案列表"],
    "pred": "模型预测答案",
    "generation_time": 1.23,
    "retrieval_results": ["检索到的片段"],
    "eval_results": {
        "em": 0.85,
        "f1": 0.92
    }
}
```

## 故障排除

### 常见问题

1. **配置文件错误**
   - 检查YAML和JSON文件格式
   - 确认配置文件路径正确
   - 验证配置参数完整性

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的GPU内存
   - 验证模型文件完整性

3. **RAG配置错误**
   - 检查RAG配置文件格式
   - 确保检索语料库已构建
   - 验证检索方法配置

4. **CUDA内存不足**
   - 减少`max_length`或`max_new_tokens`
   - 使用更小的模型
   - 减少`retrieval_topk`

5. **数据集加载失败**
   - 检查网络连接（FlashRAG会自动下载数据集）
   - 验证数据集名称是否正确
   - 确保有足够的磁盘空间

### 调试模式

在代码中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在Pipeline初始化时启用详细日志
pipeline = CausalEditorPipeline(..., verbose=True)
```

## 扩展功能

### 1. 自定义检索方法

```python
# 在config中指定不同的检索方法
config["retrieval_method"] = "dpr"  # 或 "contriever", "ance", 等
```

### 2. 添加新的评估指标

```python
config["metric"] = ["em", "f1", "bleu", "rouge"]
```

### 3. 支持的数据集

FlashRAG支持多种数据集：

```python
# 可用的数据集
supported_datasets = [
    "nq",        # Natural Questions
    "msmarco",   # MS MARCO
    "hotpotqa",  # HotpotQA
    "triviaqa",  # TriviaQA
    "webq",      # WebQuestions
    "squad",     # SQuAD
    "popqa",     # PopQA
    "arc",       # ARC
    "mmlu",      # MMLU
]

# 使用不同数据集
config["dataset_name"] = "hotpotqa"  # 多跳推理数据集
dataset = Dataset(config, "hotpotqa")
```

## 性能优化

1. **使用缓存**: 启用检索缓存以避免重复计算
2. **批处理**: 对于大数据集，考虑分批处理
3. **模型量化**: 使用量化模型减少内存使用
4. **并行处理**: 利用多GPU进行并行推理

## 参考资料

- [FlashRAG文档](https://github.com/RUC-NLPIR/FlashRAG)
- [FlashRAG支持的数据集](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/dataset.md)
- [Llama-2模型](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [RAG评估指标](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/evaluation.md)