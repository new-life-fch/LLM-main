# CausalEditor: 层级化因果溯源与反事实编辑

## 项目概述

CausalEditor 是一种新颖的LLM幻觉修正方法，通过**层级化因果溯源**和**反事实编辑**技术，在推理时精确修正大语言模型的幻觉问题。与传统的统一真实性方向编辑（如TruthX）不同，CausalEditor采用预计算的因果知识图谱激活编码，实现"外科手术式"的精确编辑。

## 核心思想

### 因果断裂检测
- **问题假设**：LLM幻觉往往源于信息流在特定层、特定神经元上的"因果断裂"
- **解决方案**：通过预计算的知识激活指纹，实时检测生成过程中的因果冲突

### 反事实编辑
- **精确编辑**：不是应用统一的真实性方向，而是基于检索到的正确知识进行精确的反事实编辑
- **最小干预**：只在检测到冲突时进行编辑，最大化保留原始模型的语言能力

## 🚀 快速开始

### 第1步：环境准备

```bash
# 解决版本兼容性问题
python fix_compatibility.py

# 或者手动创建新环境（推荐）
conda create -n causal-editor python=3.10 -y
conda activate causal-editor
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install transformers==4.30.0 faiss-cpu -c conda-forge -y
pip install numpy pandas requests tqdm scikit-learn matplotlib
```

### 第2步：测试基础功能

```bash
# 测试核心组件
python test_basic_functionality.py
```

预期输出：
```
✅ 核心组件导入成功
✅ 预计算组件导入成功
✅ 基础依赖: PyTorch, Transformers, FAISS等
✅ 配置文件加载成功
✅ 测试数据已创建
```

### 第3步：预计算知识指纹

```bash
# 从Wikidata提取知识并构建激活指纹
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-dir ./precomputed_data/llama2-7b \
    --knowledge-type wikidata \
    --limit 10000 \
    --device cuda \
    --batch-size 8
```

### 第4步：使用CausalEditor

```python
from modeling_llms.modeling_llama_causal import CausalLlamaForCausalLM
from transformers import AutoTokenizer
import torch

# 加载集成CausalEditor的模型
model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    vector_db_path="./precomputed_data/llama2-7b/vector_database",
    edit_strength=1.0,
    top_layers=10,
    device="cuda"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model.set_tokenizer(tokenizer)

# 生成无幻觉文本
question = "What year did Einstein publish his paper on the photoelectric effect?"
inputs = tokenizer(question, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"答案: {answer}")
```

### 第5步：评估效果

```bash
# 在TruthfulQA上评估
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/evaluation \
    --mode both \
    --edit-strength 1.0 \
    --top-layers 10
```

## 主要特性

### 🎯 精确编辑
- **外科手术式修正**：只修改错误的关键事实，保持语言流畅性
- **实时检测**：推理时动态检测因果冲突
- **最小副作用**：相比RAG等方法，对原文改动最小

### ⚡ 高效推理
- **低延迟**：推理时间增加<10%
- **单卡友好**：支持在单张GPU上运行
- **可扩展**：支持增量更新知识库

### 🔧 易于使用
- **即插即用**：与现有Transformers模型无缝集成
- **配置灵活**：丰富的参数调优选项
- **工具完整**：提供完整的预计算和评估工具

## 配置调优

### 关键参数说明
```python
model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    # 基础配置
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    vector_db_path="./path/to/vector_database",
    
    # 编辑强度：控制修正幅度 (0.1-3.0)
    edit_strength=1.0,
    
    # 编辑层数：参与编辑的顶层数量 (5-20)
    top_layers=10,
    
    # 相似度阈值：激活检索的最低相似度 (0.5-0.9)
    similarity_threshold=0.8,
    
    # 冲突阈值：触发编辑的冲突程度 (0.3-0.8)
    conflict_threshold=0.6
)
```

### 性能调优建议
- **编辑强度**：从1.0开始，如果修正不足可增加到1.5-2.0
- **层数选择**：模型层数的1/3左右通常效果最佳
- **阈值设置**：高阈值=高精度低召回，低阈值=高召回低精度

## 项目架构

```
CausalEditor/
├── causal_editor/                    # 核心模块
│   ├── core/                        # 核心组件
│   │   ├── causal_editor.py         # 主要CausalEditor类
│   │   ├── conflict_detector.py     # 因果冲突检测器
│   │   ├── counterfactual_editor.py # 反事实编辑器
│   │   └── vector_database.py       # 向量数据库管理
│   ├── precompute/                  # 预计算模块
│   │   ├── knowledge_extractor.py   # 知识提取
│   │   ├── fingerprint_builder.py   # 激活指纹构建
│   │   └── precompute_pipeline.py   # 预计算流水线
│   └── utils/                       # 工具函数
├── modeling_llms/                   # 集成CausalEditor的模型文件
│   └── modeling_llama_causal.py     # Llama + CausalEditor
├── scripts/                         # 脚本文件
│   ├── precompute_causal_knowledge.py  # 预计算脚本
│   └── evaluate_truthfulqa_causal.py   # 评估脚本
├── configs/                         # 配置文件
│   └── causal_editor_config.json    # 默认配置
└── test_data/                       # 测试数据
```

## 常见用例

### 用例1：科学事实修正
```python
questions = [
    "What year was the theory of relativity published?",
    "Who discovered DNA structure?", 
    "When did World War II end?"
]

for q in questions:
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=30)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
```

### 用例2：多选题评估
```python
# 设置多选题模式
model.set_generation_mode(is_mc=True, prompt_length=len(question_tokens))

# 计算答案概率
prob_true = model.get_answer_probability(question, correct_answer)
prob_false = model.get_answer_probability(question, wrong_answer)
```

### 用例3：批量处理
```python
# 批量生成（推荐方式）
questions = ["question1", "question2", "question3"]
answers = model.batch_generate(questions, max_new_tokens=50)
```

## 进阶功能

### 自定义知识源
```python
# 使用CSV文件作为知识源
python scripts/precompute_causal_knowledge.py \
    --knowledge-type csv \
    --csv-path ./custom_knowledge.csv \
    --limit 5000
```

CSV格式要求：
```csv
subject,relation,object,confidence,text
Einstein,published_paper_on,Photoelectric Effect,1.0,Einstein published a paper on the photoelectric effect.
Einstein,published_year,1905,1.0,Einstein published his paper in 1905.
```

### 增量更新
```python
# 添加新知识到现有数据库
python scripts/precompute_causal_knowledge.py \
    --incremental \
    --existing-db-path ./precomputed_data/llama2-7b/vector_database \
    --knowledge-type csv \
    --csv-path ./new_knowledge.csv \
    --limit 1000
```

### 统计监控
```python
# 获取实时统计
stats = model.get_causal_editor_statistics()
print(f"冲突检测率: {stats['conflict_detector_stats']['conflict_rate']:.3f}")
print(f"编辑成功率: {stats['counterfactual_editor_stats']['success_rate']:.3f}")
print(f"平均编辑幅度: {stats['counterfactual_editor_stats']['average_edit_magnitude']:.3f}")
```

## 问题排查

### Q: 预计算阶段显存不足？
```bash
# 降低批处理大小
python scripts/precompute_causal_knowledge.py --batch-size 4

# 或使用CPU（较慢）
python scripts/precompute_causal_knowledge.py --device cpu
```

### Q: Wikidata访问超时？
```bash
# 增加请求间隔
python scripts/precompute_causal_knowledge.py --rate-limit 2.0

# 或使用本地CSV文件
python scripts/precompute_causal_knowledge.py --knowledge-type csv --csv-path ./local_data.csv
```

### Q: 编辑效果不明显？
```python
# 增加编辑强度
model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    # ... 其他参数
    edit_strength=1.5,  # 从1.0增加到1.5
    top_layers=15       # 增加编辑层数
)
```

### Q: 推理速度较慢？
```python
# 减少编辑层数
model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    # ... 其他参数
    top_layers=5,                    # 减少层数
    similarity_threshold=0.9         # 提高阈值，减少检索
)
```

## 实验设计

### 评估指标
1. **TruthfulQA**:
   - MC1/MC2准确率（多选题）
   - True*Info评分（生成任务）

2. **效率指标**:
   - 推理延迟
   - 显存开销
   - 冲突检测率

3. **质量指标**:
   - 事实准确性改进
   - 语言流畅度保持
   - 副作用分析

### 对比方法
- **Baseline**: 原始LLM
- **TruthX**: 统一真实性方向编辑
- **RAG**: 检索增强生成
- **CausalEditor**: 本方法

## 消融实验

### RQ1: 组件有效性
```bash
# 禁用编辑，仅检测
python scripts/evaluate_truthfulqa_causal.py \
    --edit-strength 0.0  # 验证检测机制的有效性
```

### RQ2: 参数敏感性
```bash
# 不同编辑强度
for strength in 0.5 1.0 1.5 2.0; do
    python scripts/evaluate_truthfulqa_causal.py \
        --edit-strength $strength \
        --output-dir ./results/strength_$strength
done

# 不同层数
for layers in 5 10 15 20; do
    python scripts/evaluate_truthfulqa_causal.py \
        --top-layers $layers \
        --output-dir ./results/layers_$layers
done
```

### RQ3: 精准性分析
- 分析编辑的具体位置和幅度
- 对比CausalEditor与RAG的输出差异
- 评估"外科手术式"编辑的精确度

## 论文实验复现

### 复现主实验
```bash
# 1. 预计算（约2-4小时）
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --limit 10000 \
    --output-dir ./precomputed_data/llama2-7b

# 2. TruthfulQA评估（约1-2小时）  
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/truthfulqa_main \
    --mode both

# 3. 查看结果
cat ./results/truthfulqa_main/evaluation_summary.json
```

### 预期改进效果
- **TruthfulQA MC1准确率**: +15~25%
- **TruthfulQA MC2准确率**: +10~20%  
- **生成质量**: 显著减少事实性错误
- **推理延迟**: 增加<10%

## 核心组件说明

### 预计算工具
- `WikidataExtractor`: 从Wikidata提取高质量知识三元组
- `ActivationFingerprintBuilder`: 构建LLM激活指纹，支持多层并行
- `PrecomputePipeline`: 完整的预计算流水线，支持增量更新

### 核心组件
- `CausalEditor`: 主控制器，整合所有组件
- `CausalConflictDetector`: 检测因果冲突，支持实体识别和知识检索
- `CounterfactualEditor`: 执行反事实编辑，计算最优编辑方向
- `VectorDatabase`: FAISS向量数据库管理，支持GPU加速

### 模型集成
- `CausalLlamaForCausalLM`: 集成CausalEditor的Llama模型
- 支持多选题和生成任务的模式切换
- 兼容Transformers库的标准接口

## 注意事项

1. **显存需求**: 预计算阶段需要较大显存，建议24GB以上
2. **时间开销**: 首次预计算可能需要数小时，建议使用缓存
3. **模型兼容**: 目前主要支持Llama系列，其他模型需要适配
4. **数据质量**: 知识三元组的质量直接影响编辑效果

## 引用

如果您在研究中使用了CausalEditor，请引用：

```bibtex
@misc{causaleditor2024,
    title={CausalEditor: 层级化因果溯源与反事实编辑},
    author={CausalEditor Team},
    year={2024},
    note={基于层级化因果溯源的LLM幻觉修正方法}
}
``` 