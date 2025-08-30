# CLAUDE.md

本文件为Claude Code (claude.ai/code) 在使用此代码库时提供指导。

## 项目概述

这是一个 **CausalEditor** 项目 - 针对大型语言模型(LLM)的动态因果追踪和反事实编辑的高级实现。该系统旨在通过实时冲突检测和推理过程中的目标激活编辑来纠正LLM中的幻觉问题。

## 核心架构组件

### 核心模块
- **`causal_editor/core/`** - CausalEditor主要实现
  - `causal_editor.py` - 动态因果编辑的中央协调器
  - `conflict_detector.py` - 推理过程中的实时冲突检测
  - `counterfactual_editor.py` - 激活编辑和干预逻辑

### 动态处理管道
- **`causal_editor/dynamic/`** - 实时处理组件
  - `candidate_filter.py` - 实体提取和候选过滤
  - `entity_extractor.py` - 多方法实体提取(NER, spaCy, 正则表达式)
  - `fingerprint_builder.py` - 动态激活指纹生成
  - `vector_index.py` - 向量相似性搜索和索引
  - `web_knowledge_retriever.py` - 外部知识检索

### 模型集成
- **`modeling_llama_causal/`** - 集成CausalEditor的自定义Llama模型
  - `modeling_llama_causal.py` - 带有激活钩子的修改版Llama架构

## 开发命令

### 运行测试
```bash
# 带调试的主要综合测试
python test.py

# 实体提取方法比较
python test_entity.py

# 详细实体提取分析
python test_entity_extraction.py

# TruthfulQA评估
python scripts/evaluate_truthfulqa_causal.py
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 使用不同模型运行
系统支持多种模型配置。默认模型为测试用的 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`，但可以配置为更大的模型，如 `meta-llama/Llama-2-7b-hf`。

## 配置

### 主配置文件
- **`configs/causal_editor_config.json`** - 所有组件的中央配置
  - 模型设置 (device, batch_size, torch_dtype)
  - CausalEditor参数 (edit_strength, similarity_threshold, conflict_threshold)
  - 向量数据库配置 (HNSW索引)
  - 评估设置

### 关键配置参数
- `edit_strength`: 控制激活编辑的强度 (默认: 1.5)
- `similarity_threshold`: 指纹匹配阈值 (默认: 0.5)
- `conflict_threshold`: 冲突检测阈值 (默认: 0.4)
- `num_middle_layers`: 要编辑的transformer层数 (默认: 10)

## 测试输出结构

结果保存到 `result/` 目录：
- `enhanced_dynamic_causal_editor_results.json` - 详细测试结果
- `enhanced_dynamic_causal_editor_results.csv` - 用于分析的表格结果
- `enhanced_dynamic_causal_editor_report.txt` - 人类可读的测试报告
- `debug_details.json` - 调试信息和统计数据

## 代码库使用指南

### 实体提取方法
系统支持多种实体提取方法：
- `HYBRID` - 结合多种方法以获得最佳结果
- `NER_TRANSFORMERS` - 使用基于transformer的NER模型
- `NER_SPACY` - 使用spaCy的NER管道
- `REGEX_PATTERN` - 基于规则的模式匹配

### 缓存管理
- **`cache/web_knowledge/`** - 缓存的知识检索结果
- 知识缓存存储在SQLite数据库中 (`knowledge_cache.db`)

### 动态模式操作
CausalEditor在推理过程中实时运行：
1. **实体提取** - 识别输入文本中的实体
2. **指纹生成** - 为实体创建激活指纹
3. **冲突检测** - 与已知事实模式进行比较
4. **激活编辑** - 当检测到冲突时修改模型激活

## 重要开发说明

- 该系统设计用于 **防御性研究目的** - 研究和缓解LLM幻觉问题
- 模型需要CUDA以获得最佳性能；提供CPU备用方案
- 可以禁用网络知识检索进行离线测试
- 代码库包含广泛的调试和监控功能
- 所有测试脚本都包含全面的错误处理和进度报告