# CausalEditor: 动态因果追踪与反事实编辑系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

CausalEditor 是一个针对大型语言模型(LLM)的高级动态因果追踪和反事实编辑系统。该系统旨在通过实时冲突检测和推理过程中的目标激活编辑来纠正LLM中的幻觉问题，并集成了先进的RAG(检索增强生成)技术以提升知识检索和编辑精度。

## 🌟 核心特性

### 🔍 RAG增强检索系统特性
- **大规模文档检索**: 集成BGE模型和FAISS HNSW索引，支持千万级文档的高效检索
- **智能重排序**: 使用BAAI/bge-reranker-large进行精确重排序
- **智能Fallback机制**: 基于相关性分数的动态网络搜索回退
- **高性能向量化**: 支持GPU加速的向量检索和内存优化

### 🧠 知识图谱系统特性
- **多方法实体提取**: 支持NER、spaCy、正则表达式、混合模式等多种提取策略
- **结构化知识获取**: 从维基百科、Wikidata等获取结构化三元组
- **关系推理**: 基于实体关系的智能推理和验证
- **知识图谱缓存**: 高效的本地知识缓存和管理

### ⚡ 统一因果编辑核心
- **实时激活指纹构建**: 将知识信息转换为模型特定的激活指纹
- **精确冲突检测**: 在推理过程中实时检测模型生成与事实知识的冲突
- **外科手术式编辑**: 仅在检测到冲突时进行最小化激活修正
- **多层协同编辑**: 支持多个transformer层的协同编辑策略
- **动态阈值调整**: 基于统计数据的自适应阈值优化

## 🏗️ 系统架构

CausalEditor 支持两套独立的知识检索和编辑系统，可根据需求灵活选择：

### 🔍 系统一：RAG增强检索系统
```
输入文本 → RAG检索 → 文档重排序 → 知识融合 → 指纹构建 → 向量索引
                                                              ↓
LLM推理 ← 反事实编辑 ← 冲突检测 ← 相似度搜索 ← 动态阈值判断 ←────────┘
```

### 🧠 系统二：实体提取知识图谱系统
```
输入文本 → 实体提取 → 获取三元组 → 知识验证 → 指纹构建 → 向量索引
                                                              ↓
LLM推理 ← 反事实编辑 ← 冲突检测 ← 相似度搜索 ← 动态阈值判断 ←────────┘
```

### 🔄 系统选择模式
- **`rag_only`**: 仅使用RAG检索系统，适合大规模文档检索
- **`kg_only`**: 仅使用知识图谱系统，适合结构化知识推理
- **`hybrid`**: 混合使用两套系统，获得最佳效果

### 核心组件结构

```
New_Project-CausalEdit/
├── causal_editor/                    # 核心编辑器模块
│   ├── core/                        # 核心功能
│   │   ├── causal_editor.py         # 主协调器
│   │   ├── conflict_detector.py     # 冲突检测器
│   │   └── counterfactual_editor.py # 反事实编辑器
│   ├── dynamic/                     # 动态处理管道
│   │   ├── entity_extractor.py      # 实体提取
│   │   ├── rag_retriever.py         # RAG检索器
│   │   ├── rag_candidate_filter.py  # RAG候选过滤
│   │   ├── fingerprint_builder.py   # 指纹构建
│   │   ├── vector_index.py          # 向量索引
│   │   └── web_knowledge_retriever.py # 网络知识检索
│   └── utils/                       # 工具模块
├── modeling_llama_causal/           # 模型集成
│   ├── modeling_llama_causal.py     # 基础Llama集成
│   └── modeling_llama_causal_Llama2.py # Llama2专用集成
├── configs/                         # 配置文件
│   ├── causal_editor_config.json    # 主配置
│   └── rag_config_example.json      # RAG配置示例
├── tests/                          # 测试套件
├── scripts/                        # 评估脚本
└── docs/                          # 文档
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (推荐)
- **内存**: 16GB+ RAM
- **显存**: 8GB+ VRAM (用于7B模型)

### 安装依赖

```bash
cd New_Project-CausalEdit
pip install -r requirements.txt
```

**首次使用需要下载模型**:
```bash
# 下载spaCy模型
python -m spacy download en_core_web_sm

# 下载Wikipedia数据(可选，用于RAG)
python tests/download_wikipedia.py
```

### 基本使用

#### 1. 使用预配置模型

```python
from modeling_llama_causal.modeling_llama_causal_Llama2 import CausalLlama2ForCausalLM
from transformers import AutoTokenizer

# 加载集成CausalEditor的模型
model = CausalLlama2ForCausalLM.from_pretrained_with_dynamic_causal_editor(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    edit_strength=1.5,
    similarity_threshold=0.5,
    use_rag_retrieval=True
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 准备输入
input_text = "The American Civil War ended in 1975"
model.causal_editor.prepare_for_input(input_text)

# 生成回复（自动进行因果编辑）
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 2. 使用RAG检索系统

```python
from causal_editor.core.causal_editor import CausalEditor

# 初始化RAG检索系统
editor = CausalEditor(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    edit_strength=1.5,
    num_middle_layers=10,
    similarity_threshold=0.5,
    conflict_threshold=0.4,
    use_rag_retrieval=True,
    retrieval_mode="rag_only",  # 仅使用RAG检索
    device="cuda"
)

# 为输入准备RAG检索
input_text = "Paris is the capital of Germany"
editor.prepare_for_input(input_text)

# 在推理过程中进行编辑
edited_activations = editor.edit_activations(
    activations=current_activations,
    generated_tokens=["Paris", "is"],
    input_text=input_text
)
```

#### 3. 使用知识图谱系统

```python
from causal_editor.core.causal_editor import CausalEditor

# 初始化知识图谱系统
editor = CausalEditor(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    edit_strength=1.5,
    num_middle_layers=10,
    similarity_threshold=0.5,
    conflict_threshold=0.4,
    use_rag_retrieval=False,
    retrieval_mode="kg_only",  # 仅使用知识图谱
    device="cuda"
)

# 为输入准备实体提取和三元组获取
input_text = "Einstein was born in Germany"
editor.prepare_for_input(input_text)

# 在推理过程中进行编辑
edited_activations = editor.edit_activations(
    activations=current_activations,
    generated_tokens=["Einstein", "was"],
    input_text=input_text
)
```

#### 4. 配置文件使用

```python
# 使用混合模式配置
editor = CausalEditor.from_config(
    config_path="configs/causal_editor_config.json",
    model=model,
    tokenizer=tokenizer
)

# 使用纯RAG系统配置
editor_rag = CausalEditor.from_config(
    config_path="configs/rag_only_config.json",
    model=model,
    tokenizer=tokenizer
)

# 使用纯知识图谱系统配置
editor_kg = CausalEditor.from_config(
    config_path="configs/kg_only_config.json",
    model=model,
    tokenizer=tokenizer
)
```

## 🧪 运行测试

### 快速测试

```bash
# 检查环境
./run_tests.sh check

# 运行基础测试
./run_tests.sh quick

# 运行完整测试套件
./run_tests.sh full

# 运行RAG系统测试
./run_tests.sh rag

# 运行性能基准测试
./run_tests.sh benchmark
```

### 主要测试脚本

```bash
# 综合功能测试（推荐）
python tests/test.py

# RAG系统测试
python tests/test_rag_system.py

# 实体提取方法比较
python tests/test_entity_extraction.py

# TruthfulQA评估
python scripts/evaluate_truthfulqa_causal.py

# 性能基准测试
python tests/benchmark_rag.py
```

## ⚙️ 配置说明

### 核心配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `edit_strength` | float | 1.5 | 激活编辑强度，范围0.5-3.0 |
| `similarity_threshold` | float | 0.5 | 指纹匹配阈值，范围0.0-1.0 |
| `conflict_threshold` | float | 0.4 | 冲突检测阈值，范围0.0-1.0 |
| `num_middle_layers` | int | 10 | 参与编辑的transformer层数 |
| `use_rag_retrieval` | bool | true | 是否启用RAG检索 |
| `retrieval_mode` | string | "rag_enhanced" | 检索模式："rag_only", "kg_only", "hybrid" |
| `max_candidates` | int | 50 | 最大候选三元组数量 |

### RAG系统配置

```json
{
  "rag_retrieval": {
    "enabled": true,
    "model_name": "BAAI/bge-large-en-v1.5",
    "top_k": 10,
    "min_score": 0.3
  },
  "reranker_config": {
    "enabled": true,
    "model_name": "BAAI/bge-reranker-large",
    "initial_candidates": 1000,
    "final_top_k": 10
  },
  "fallback_config": {
    "enabled": true,
    "threshold_high": 0.35,
    "enable_dynamic_threshold": true
  }
}
```

### 知识图谱系统配置

```json
{
  "knowledge_extraction": {
    "entity_extraction_method": "HYBRID",
    "web_knowledge_retrieval": "HYBRID",
    "max_entities_per_input": 10,
    "enable_caching": true
  },
  "web_knowledge_retrieval": {
    "strategy": "HYBRID",
    "wikipedia_enabled": true,
    "wikidata_enabled": true,
    "timeout": 30
  }
}
```

### 实体提取方法

- `HYBRID` - 结合多种方法获得最佳效果 (推荐)
- `NER_TRANSFORMERS` - 基于transformer的NER模型
- `NER_SPACY` - 使用spaCy的NER管道
- `REGEX_PATTERN` - 基于规则的模式匹配

### 网络检索策略

- `WIKIPEDIA_API` - Wikipedia API检索
- `WIKIDATA_API` - Wikidata结构化检索
- `HYBRID` - 混合策略 (推荐)

## 📊 结果输出

测试结果保存在 `result/` 目录：

```
result/
├── enhanced_dynamic_causal_editor_results.json  # 详细测试结果
├── enhanced_dynamic_causal_editor_results.csv   # 表格格式结果
├── enhanced_dynamic_causal_editor_report.txt    # 人类可读报告
├── debug_details.json                          # 调试信息
└── test_results/                               # 各类测试结果
    ├── rag_upgrade_test_results.json
    └── intermediate_results_*.json
```

## 🔧 高级用法

### 自定义模型集成

```python
# 使用自定义模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

editor = CausalEditor(
    model=model,
    tokenizer=tokenizer,
    model_name="your-model",
    use_dynamic_mode=True,
    use_rag_retrieval=True
)
```

### RAG系统优化

```python
# 配置高性能RAG
rag_config = {
    "rag_retrieval": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "top_k": 20,
        "enable_gpu": True,
        "use_fp16": True
    },
    "index_config": {
        "index_type": "HNSW",
        "hnsw_m": 64,
        "hnsw_ef_construction": 512,
        "enable_sharding": True
    }
}

editor = CausalEditor.from_config(rag_config, model, tokenizer)
```

### 批处理和性能优化

```python
# 启用批处理优化
editor = CausalEditor(
    model_name="meta-llama/Llama-2-7b-hf",
    batch_size=16,
    enable_memory_mapping=True,
    max_memory_usage=0.8
)

# 批量预处理
inputs = ["Paris is the capital", "Einstein discovered", "Python is a"]
for text in inputs:
    editor.prepare_for_input(text)
```

### 统计和监控

```python
# 获取详细统计
stats = editor.get_statistics()
print(f"冲突检测次数: {stats['conflict_detector_stats']['detection_count']}")
print(f"成功编辑次数: {stats['counterfactual_editor_stats']['successful_edits']}")
print(f"RAG检索统计: {stats['rag_retrieval_stats']}")

# 获取RAG性能统计
rag_stats = editor.rag_retriever.get_score_statistics()
print(f"平均相关性分数: {rag_stats['avg_score']}")
print(f"Fallback触发率: {rag_stats['fallback_triggered']}")
```

## 💡 工作原理

### 🔍 RAG检索系统工作流程
```
输入: "Paris is the capital of Germany"
↓
RAG检索: 从千万级文档中检索相关段落
↓
重排序: BGE-reranker筛选top-10最相关结果
↓
知识融合: 生成结构化知识表示
↓
指纹构建: [layer_0: tensor(...), layer_1: tensor(...), ...]
↓
向量索引: 存储到FAISS HNSW索引
```

### 🧠 知识图谱系统工作流程
```
输入: "Einstein was born in Germany"
↓
实体提取: ["Einstein", "Germany"]
↓
三元组获取: 从Wikipedia/Wikidata获取结构化知识
↓
知识验证: [("Einstein", "born_in", "Germany"), ...]
↓
指纹构建: [layer_0: tensor(...), layer_1: tensor(...), ...]
↓
向量索引: 存储到FAISS HNSW索引
```

### ⚡ 统一因果编辑流程
```
推理过程: 模型生成文本
↓
激活比较: 当前激活 vs 预存指纹
↓
冲突检测: 相似度 < threshold → 检测到冲突
↓
精确编辑: 原始激活 + 编辑向量 × 强度 = 修正激活
```

## 🔬 研究背景与创新点

### 核心创新
1. **双系统架构设计**: 创新性地设计了RAG检索和知识图谱两套独立且可协同的系统
2. **RAG-增强的因果编辑**: 首次将大规模RAG检索与因果编辑结合
3. **结构化知识推理**: 基于实体提取和三元组的精确知识推理
4. **统一因果编辑框架**: 两套系统共享统一的指纹构建和冲突检测机制
5. **动态阈值调整**: 基于统计反馈的自适应阈值优化
6. **智能系统切换**: 支持rag_only、kg_only、hybrid三种模式的灵活切换

### 技术优势
- **高精度**: BGE模型+重排序确保检索质量
- **高效率**: HNSW索引支持毫秒级检索
- **可扩展**: 支持千万级文档的知识库
- **自适应**: 动态调整确保最优性能

## 🧪 评估与基准

### 支持的评估数据集
- **TruthfulQA**: 事实准确性评估
- **自定义幻觉测试集**: 涵盖历史、科学、地理等领域
- **多难度级别**: Easy/Medium/Hard分级测试

### 性能指标
- **幻觉纠正率**: 成功纠正错误事实的比例
- **检索精度**: RAG系统的检索准确性
- **编辑成功率**: 因果编辑的成功比例
- **响应时间**: 端到端处理延迟

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：

- 如何提交bug报告和功能请求
- 代码风格指南和测试要求
- 如何贡献新的实体提取方法
- 如何添加新的知识检索源

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd New_Project-CausalEdit

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8

# 运行测试
python -m pytest tests/

# 代码格式化
black causal_editor/
flake8 causal_editor/
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## ⚠️ 注意事项

- **研究用途**: 该系统设计用于防御性研究目的，旨在研究和缓解LLM幻觉问题
- **硬件要求**: 建议使用GPU以获得最佳性能，支持CPU备用方案
- **内存需求**: 大型模型和RAG索引需要足够的内存和显存
- **网络连接**: RAG检索和Fallback功能需要稳定的网络连接
- **首次运行**: 可能需要下载较大的模型文件和数据集

## 🔧 故障排除

### 常见问题

#### 1. 内存不足
```bash
# 减少批处理大小
"batch_size": 8

# 启用内存映射
"enable_memory_mapping": true

# 使用较小模型
"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

#### 2. RAG检索速度慢
```bash
# 调整HNSW参数
"hnsw_ef_search": 128

# 启用GPU加速
"enable_gpu": true

# 减少候选数
"initial_candidates": 500
```

#### 3. 模型加载失败
```bash
# 检查CUDA版本
nvidia-smi

# 使用CPU模式
"device": "cpu"

# 检查模型权限
huggingface-cli login
```

## 📞 支持与反馈

如有问题或建议，请：

1. 查看 [docs/](docs/) 目录中的详细文档
2. 运行 `./run_tests.sh check` 检查环境
3. 提交 [GitHub Issue](https://github.com/your-repo/issues)
4. 查看 [FAQ](docs/FAQ.md) 常见问题

## 🏆 致谢

感谢以下项目和团队的贡献：

- **Hugging Face**: Transformers库和模型托管
- **BAAI**: BGE嵌入模型和重排序模型
- **Facebook Research**: FAISS向量检索库
- **spaCy**: 自然语言处理工具
- **Wikipedia**: 开放知识数据源

特别感谢所有为此项目做出贡献的研究者和开发者。

---

**免责声明**: 本工具仅用于研究目的。请负责任地使用，遵守相关法律法规和伦理准则。在使用过程中请注意保护用户隐私和数据安全。