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
└── data/                           # 数据文件
```

## 使用方法

### 1. 环境安装

```bash
# 安装依赖
pip install torch transformers faiss-cpu numpy pandas requests tqdm

# 对于GPU加速的FAISS（可选）
# pip install faiss-gpu
```

### 2. 预计算阶段

#### 方法一：从Wikidata提取知识
```bash
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-dir ./precomputed_data/llama2-7b \
    --knowledge-type wikidata \
    --limit 10000 \
    --device cuda \
    --batch-size 8
```

#### 方法二：从CSV文件构建
```bash
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-dir ./precomputed_data/llama2-7b \
    --knowledge-type csv \
    --csv-path ./data/knowledge_triplets.csv \
    --limit 5000
```

#### 预计算输出
预计算完成后，会在输出目录生成：
- `vector_database/` - FAISS向量数据库
- `knowledge_triplets.json` - 提取的知识三元组
- `activation_fingerprints.npz` - 激活指纹
- `statistics.json` - 统计信息
- `precompute_report.txt` - 可读报告

### 3. 使用CausalEditor

#### 基本使用
```python
from modeling_llms.modeling_llama_causal import CausalLlamaForCausalLM
from transformers import AutoTokenizer
import torch

# 加载模型和tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
vector_db_path = "./precomputed_data/llama2-7b/vector_database"

model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    model_name_or_path=model_name,
    vector_db_path=vector_db_path,
    edit_strength=1.0,
    top_layers=10,
    device="cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model.set_tokenizer(tokenizer)

# 生成文本
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

#### 配置文件使用
```python
# 使用配置文件
from causal_editor import CausalEditor

config_path = "configs/causal_editor_config.json"
causal_editor = CausalEditor.from_config(config_path, vector_db_path)
```

### 4. 评估

#### TruthfulQA评估
```bash
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/truthfulqa_causal \
    --mode both \
    --edit-strength 1.0 \
    --top-layers 10
```

#### 评估模式
- `mc`: 仅多选题评估
- `generation`: 仅开放式生成评估  
- `both`: 完整评估（默认）

## 关键参数说明

### CausalEditor参数
- `edit_strength`: 编辑强度，控制修正的幅度（默认：1.0）
- `top_layers`: 参与编辑的顶层数量（默认：10）
- `similarity_threshold`: 激活相似度阈值（默认：0.8）
- `conflict_threshold`: 冲突检测阈值（默认：0.6）

### 预计算参数
- `limit`: 提取的知识三元组数量限制
- `target_layers`: 目标层列表，None表示所有层
- `batch_size`: 批处理大小
- `rate_limit`: API请求速率限制（Wikidata）

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

# 使用随机向量编辑
# 通过修改counterfactual_editor.py实现
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

## 统计信息获取

```python
# 获取运行时统计
stats = model.get_causal_editor_statistics()
print(f"冲突检测次数: {stats['conflict_detector_stats']['detection_count']}")
print(f"冲突发现率: {stats['conflict_detector_stats']['conflict_rate']:.3f}")
print(f"编辑成功率: {stats['counterfactual_editor_stats']['success_rate']:.3f}")

# 重置统计
model.reset_causal_editor_statistics()
```

## 高级功能

### 增量更新数据库
```bash
python scripts/precompute_causal_knowledge.py \
    --incremental \
    --existing-db-path ./precomputed_data/llama2-7b/vector_database \
    --limit 1000
```

### 自定义知识源
```python
# 实现自定义知识提取器
from causal_editor.precompute.knowledge_extractor import KnowledgeExtractor

class CustomExtractor(KnowledgeExtractor):
    def extract_triplets(self, limit: int = 10000):
        # 自定义提取逻辑
        return triplets
    
    def get_high_frequency_relations(self, top_k: int = 1000):
        return relations
```

### 可视化分析
```python
# 分析激活空间变化
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 获取编辑前后的激活状态
# 使用t-SNE可视化
```

## 文件说明

### 核心组件
- `CausalEditor`: 主控制器，整合所有组件
- `CausalConflictDetector`: 检测因果冲突，支持实体识别和知识检索
- `CounterfactualEditor`: 执行反事实编辑，计算最优编辑方向
- `VectorDatabase`: FAISS向量数据库管理，支持GPU加速

### 预计算工具
- `WikidataExtractor`: 从Wikidata提取高质量知识三元组
- `ActivationFingerprintBuilder`: 构建LLM激活指纹，支持多层并行
- `PrecomputePipeline`: 完整的预计算流水线，支持增量更新

### 模型集成
- `CausalLlamaForCausalLM`: 集成CausalEditor的Llama模型
- 支持多选题和生成任务的模式切换
- 兼容Transformers库的标准接口

## 注意事项

1. **显存需求**: 预计算阶段需要较大显存，建议24GB以上
2. **时间开销**: 首次预计算可能需要数小时，建议使用缓存
3. **模型兼容**: 目前主要支持Llama系列，其他模型需要适配
4. **数据质量**: 知识三元组的质量直接影响编辑效果

## 故障排除

### 常见问题

1. **FAISS安装问题**
```bash
# 如果faiss-cpu安装失败
conda install -c conda-forge faiss-cpu
```

2. **显存不足**
```bash
# 降低批处理大小
python scripts/precompute_causal_knowledge.py --batch-size 4

# 使用CPU模式
python scripts/precompute_causal_knowledge.py --device cpu
```

3. **Wikidata访问限制**
```bash
# 增加请求间隔
python scripts/precompute_causal_knowledge.py --rate-limit 2.0
```

## 论文实验复现

### 完整实验流程
```bash
# 1. 预计算（Llama-2-7B）
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --limit 10000 \
    --output-dir ./precomputed_data/llama2-7b

# 2. TruthfulQA评估
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/truthfulqa_main

# 3. 消融实验
./scripts/run_ablation_studies.sh

# 4. 泛化性评估
python scripts/evaluate_nq_trivia.py
```

### 预期结果
- **TruthfulQA MC1**: 预期提升15-25%
- **推理延迟**: 增加<10%
- **编辑精准性**: 平均每个回答修正1-2个关键事实

## 致谢

本项目基于TruthfulQA数据集和相关评估工具，参考了TruthX等先行工作的思路，在此表示感谢。

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