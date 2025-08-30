# CausalEditor - 大语言模型因果编辑框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 项目简介

CausalEditor 是一个先进的大语言模型因果编辑框架，专门用于在推理过程中实时检测和修正模型的因果冲突。该框架采用动态模式，能够：

- 🔍 **实时冲突检测**：通过激活监测和动态指纹比对检测因果断裂点
- ✏️ **精确状态编辑**：基于反事实推理执行精确的激活状态编辑
- 🧠 **动态知识整合**：实时筛选和整合相关知识三元组
- 📊 **性能监控**：提供详细的统计信息和调试支持

## 🏗️ 系统架构

```
CausalEditor 框架
├── 核心组件
│   ├── CausalEditor (主控制器)
│   ├── CausalConflictDetector (冲突检测器)
│   └── CounterfactualEditor (反事实编辑器)
├── 动态模块
│   ├── DynamicCandidateFilter (候选过滤器)
│   ├── DynamicFingerprintBuilder (指纹构建器)
│   └── DynamicVectorIndex (向量索引)
└── 工具模块
    ├── EntityExtractionManager (实体提取)
    └── WebKnowledgeRetriever (网络知识获取)
```

## 📋 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (推荐，用于GPU加速)
- 内存：至少8GB RAM
- 存储：至少2GB可用空间

## 🛠️ 安装指南

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/CausalEditor.git
cd CausalEditor
```

### 2. 创建虚拟环境

```bash
# 使用conda
conda create -n causal_editor python=3.8
conda activate causal_editor

# 或使用venv
python -m venv causal_editor
source causal_editor/bin/activate  # Linux/Mac
# 或
causal_editor\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装可选依赖（用于高级功能）
pip install -r requirements-optional.txt
```

### 4. 验证安装

```bash
python test.py
```

## 🚀 快速开始

### 基础使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_editor import CausalEditor

# 1. 加载模型和分词器
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 创建因果编辑器
editor = CausalEditor(
    model=model,
    tokenizer=tokenizer,
    use_dynamic_mode=True,  # 启用动态模式
    debug_mode=True         # 启用调试模式
)

# 3. 准备输入并执行推理
input_text = "What is the capital of France?"
editor.prepare_for_input(input_text)  # 准备动态候选知识

# 4. 生成文本（自动进行冲突检测和编辑）
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

# 5. 解码结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"生成结果: {response}")

# 6. 查看统计信息
stats = editor.get_statistics()
print(f"检测冲突: {stats['conflict_detector_stats']['conflict_count']}")
print(f"执行编辑: {stats['counterfactual_editor_stats']['edit_count']}")
```

### 高级配置示例

```python
# 自定义配置
editor = CausalEditor(
    model=model,
    tokenizer=tokenizer,
    target_layers=[8, 12, 16],      # 指定编辑层
    similarity_threshold=0.9,        # 相似度阈值
    conflict_threshold=0.8,          # 冲突判定阈值
    edit_strength=0.8,               # 编辑强度
    min_confidence=0.7,              # 最小置信度
    use_dynamic_mode=True
)

# 配置动态模块
editor.dynamic_candidate_filter.set_web_retrieval_strategy('balanced')
editor.dynamic_candidate_filter.set_entity_extraction_method('spacy')

# 运行时更新配置
editor.update_config(
    similarity_threshold=0.85,
    edit_strength=1.2
)
```

## 📖 核心功能

### 1. 动态冲突检测

框架在生成过程中实时监测激活状态，通过以下机制检测因果冲突：

- **激活指纹匹配**：将当前激活与知识库中的指纹进行比对
- **语义一致性检查**：验证生成内容与已知事实的一致性
- **时序逻辑验证**：检查时间相关的因果关系

### 2. 精确状态编辑

当检测到冲突时，框架执行精确的激活状态编辑：

- **误差投影计算**：计算错误激活在正确方向上的投影
- **加权目标构建**：基于置信度构建目标激活向量
- **自适应强度调整**：根据冲突严重程度调整编辑强度

### 3. 动态知识整合

框架支持多源知识整合：

- **本地知识库**：预构建的结构化知识三元组
- **网络知识获取**：实时从网络获取相关知识
- **实体关系推断**：基于上下文推断潜在关系

## 🔧 配置选项

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `similarity_threshold` | 0.85 | 指纹相似度阈值 |
| `conflict_threshold` | 0.7 | 冲突判定阈值 |
| `edit_strength` | 1.0 | 编辑强度系数 |
| `min_confidence` | 0.6 | 最小置信度阈值 |
| `use_dynamic_mode` | True | 是否启用动态模式 |

### 动态模块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_candidates` | 50 | 最大候选数量 |
| `cache_size` | 1000 | 缓存大小 |
| `max_batch_size` | 8 | 最大批处理大小 |
| `index_type` | 'flat' | 向量索引类型 |

## 📊 性能监控

### 统计信息

```python
stats = editor.get_statistics()

# 冲突检测统计
conflict_stats = stats['conflict_detector_stats']
print(f"检测次数: {conflict_stats['detection_count']}")
print(f"冲突次数: {conflict_stats['conflict_count']}")
print(f"平均置信度: {conflict_stats['avg_confidence']}")

# 编辑统计
edit_stats = stats['counterfactual_editor_stats']
print(f"编辑次数: {edit_stats['edit_count']}")
print(f"成功编辑: {edit_stats['successful_edits']}")
print(f"平均编辑强度: {edit_stats['avg_edit_strength']}")

# 动态模块统计
filter_stats = stats['dynamic_candidate_filter_stats']
print(f"候选筛选: {filter_stats['filter_count']}")
print(f"缓存命中率: {filter_stats['cache_hit_rate']}")
```

### 性能优化建议

1. **批处理优化**：调整 `max_batch_size` 以平衡速度和内存使用
2. **缓存策略**：增大 `cache_size` 以提高缓存命中率
3. **层选择**：选择关键层进行编辑以减少计算开销
4. **阈值调优**：根据任务特点调整各种阈值参数

## 🧪 测试和验证

### 运行测试套件

```bash
# 运行完整测试
python test.py

# 运行特定类别测试
python test.py --category "历史事实测试"

# 运行调试模式
python test.py --debug
```

### 测试类别

框架包含多种测试场景：

- **历史事实测试**：验证历史事件的准确性
- **地理知识测试**：测试地理信息的正确性
- **科学事实测试**：检验科学知识的准确性
- **数学计算测试**：验证数学推理能力
- **逻辑推理测试**：测试逻辑一致性

## 📁 项目结构

```
CausalEditor/
├── causal_editor/              # 主要源代码
│   ├── __init__.py
│   ├── main.py                 # 主控制器
│   ├── core/                   # 核心组件
│   │   ├── conflict_detector.py
│   │   └── counterfactual_editor.py
│   ├── dynamic/                # 动态模块
│   │   ├── candidate_filter.py
│   │   ├── fingerprint_builder.py
│   │   └── vector_index.py
│   └── utils/                  # 工具模块
│       ├── entity_extraction.py
│       └── web_retrieval.py
├── data/                       # 数据文件
│   └── knowledge_base/
├── results/                    # 测试结果
├── test.py                     # 测试脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 开发规范

- 遵循 PEP 8 代码风格
- 添加适当的文档字符串
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 Hugging Face Transformers 团队提供的优秀框架
- 感谢 PyTorch 团队的深度学习支持
- 感谢所有贡献者的宝贵建议和代码贡献

## 📞 联系我们

- 项目主页：[GitHub Repository](https://github.com/your-repo/CausalEditor)
- 问题反馈：[Issues](https://github.com/your-repo/CausalEditor/issues)
- 邮箱：your-email@example.com

## 🔗 相关资源

- [API 文档](api_documentation.md)
- [开发者指南](developer_guide.md)
- [常见问题](faq.md)
- [更新日志](changelog.md)

---

**CausalEditor** - 让大语言模型的因果推理更加准确可靠！ 🎯