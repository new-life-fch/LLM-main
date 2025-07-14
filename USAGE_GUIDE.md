# CausalEditor 快速使用指南

## 快速开始（3步骤）

### 第1步：预计算知识指纹
```bash
# 从Wikidata提取10000个知识三元组并构建激活指纹
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-dir ./precomputed_data/llama2-7b \
    --knowledge-type wikidata \
    --limit 10000 \
    --device cuda
```

### 第2步：使用CausalEditor
```python
from modeling_llms.modeling_llama_causal import CausalLlamaForCausalLM
from transformers import AutoTokenizer

# 加载集成CausalEditor的模型
model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    vector_db_path="./precomputed_data/llama2-7b/vector_database",
    edit_strength=1.0,
    top_layers=10
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model.set_tokenizer(tokenizer)

# 生成无幻觉文本
question = "What year did Einstein publish his paper on the photoelectric effect?"
inputs = tokenizer(question, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(answer)  # 应该输出：1905年
```

### 第3步：评估效果
```bash
# 在TruthfulQA上评估
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/evaluation \
    --mode both
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

## 实验复现

### 复现论文主实验
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

---

更多详细信息请参考 [README_CausalEditor.md](./README_CausalEditor.md) 