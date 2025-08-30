# 贡献指南 (Contributing Guide)

感谢您对 CausalEditor 项目的兴趣！我们欢迎各种形式的贡献，包括但不限于代码、文档、测试和反馈。

## 📋 目录

- [开发环境设置](#开发环境设置)
- [代码贡献流程](#代码贡献流程)
- [代码规范](#代码规范)
- [测试指南](#测试指南)
- [文档贡献](#文档贡献)
- [问题报告](#问题报告)
- [功能请求](#功能请求)
- [发布流程](#发布流程)

## 🛠️ 开发环境设置

### 1. Fork 和 Clone

```bash
# Fork 项目到您的 GitHub 账户
# 然后 clone 到本地
git clone https://github.com/your-username/CausalEditor.git
cd CausalEditor

# 添加上游仓库
git remote add upstream https://github.com/original-repo/CausalEditor.git
```

### 2. 环境配置

```bash
# 创建虚拟环境
python -m venv causal_editor_env
source causal_editor_env/bin/activate  # Linux/Mac
# 或
causal_editor_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

# 安装 spaCy 模型
python -m spacy download en_core_web_sm
```

### 3. 验证安装

```bash
# 运行基本导入测试
python -c "from causal_editor import CausalEditor; print('安装成功')"

# 运行简单测试
python test_basic.py
```

## 🔄 代码贡献流程

### 1. 创建分支

```bash
# 确保主分支是最新的
git checkout main
git pull upstream main

# 创建功能分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/issue-number-description
```

### 2. 开发和提交

```bash
# 进行您的更改
# ...

# 运行测试确保没有破坏现有功能
python -m pytest tests/

# 运行代码格式检查
black causal_editor/
flake8 causal_editor/

# 提交更改
git add .
git commit -m "feat: 添加新的实体提取方法"
```

### 3. 提交 Pull Request

```bash
# 推送到您的 fork
git push origin feature/your-feature-name

# 在 GitHub 上创建 Pull Request
```

## 📝 代码规范

### Python 代码风格

我们遵循 [PEP 8](https://pep8.org/) 代码风格指南：

#### 1. 导入规范

```python
# 标准库导入
import os
import sys
from typing import Dict, List, Optional

# 第三方库导入
import torch
import numpy as np
from transformers import AutoModel

# 本地导入
from .core.causal_editor import CausalEditor
from ..utils.helper import process_data
```

#### 2. 命名规范

```python
# 类名：PascalCase
class CausalEditor:
    pass

# 函数和变量名：snake_case
def extract_entities(text: str) -> List[str]:
    entity_list = []
    return entity_list

# 常量：UPPER_SNAKE_CASE
MAX_CANDIDATES = 50
DEFAULT_THRESHOLD = 0.5
```

#### 3. 文档字符串

使用 Google 风格的文档字符串：

```python
def edit_activations(
    self,
    activations: torch.Tensor,
    generated_tokens: Optional[List[str]] = None,
    input_text: Optional[str] = None,
) -> torch.Tensor:
    """
    编辑激活状态的主入口函数
    
    Args:
        activations (torch.Tensor): 当前激活状态，形状为 [batch_size, seq_len, hidden_dim]
        generated_tokens (Optional[List[str]]): 已生成的tokens列表
        input_text (Optional[str]): 原始输入文本
        
    Returns:
        torch.Tensor: 编辑后的激活状态，形状与输入相同
        
    Raises:
        ValueError: 当activations维度不正确时
        RuntimeError: 当模型未正确初始化时
        
    Example:
        >>> editor = CausalEditor()
        >>> edited = editor.edit_activations(activations, ["hello", "world"])
    """
    pass
```

#### 4. 类型提示

始终使用类型提示：

```python
from typing import Dict, List, Optional, Union, Tuple, Any

def process_candidates(
    entities: List[str],
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """处理候选实体"""
    pass
```

### 代码质量工具

我们使用以下工具维护代码质量：

```bash
# 代码格式化
black causal_editor/ --line-length 88

# 导入排序
isort causal_editor/

# 代码检查
flake8 causal_editor/ --max-line-length=88
pylint causal_editor/

# 类型检查
mypy causal_editor/
```

## 🧪 测试指南

### 测试结构

```
tests/
├── unit/                 # 单元测试
│   ├── test_causal_editor.py
│   ├── test_conflict_detector.py
│   └── test_entity_extractor.py
├── integration/          # 集成测试
│   ├── test_full_pipeline.py
│   └── test_model_integration.py
├── performance/          # 性能测试
│   └── test_benchmarks.py
└── fixtures/            # 测试数据
    ├── sample_texts.py
    └── mock_models.py
```

### 编写测试

#### 单元测试示例

```python
import pytest
import torch
from causal_editor.core.causal_editor import CausalEditor

class TestCausalEditor:
    """CausalEditor单元测试"""
    
    @pytest.fixture
    def editor(self):
        """创建测试用的CausalEditor实例"""
        return CausalEditor(
            model_name="test-model",
            edit_strength=1.0,
            device="cpu"
        )
    
    def test_initialization(self, editor):
        """测试初始化"""
        assert editor.edit_strength == 1.0
        assert editor.device.type == "cpu"
        
    def test_prepare_for_input(self, editor):
        """测试输入准备"""
        text = "Paris is the capital of France"
        result = editor.prepare_for_input(text)
        assert result is not None
        
    @pytest.mark.parametrize("edit_strength", [0.5, 1.0, 2.0])
    def test_different_edit_strengths(self, edit_strength):
        """测试不同编辑强度"""
        editor = CausalEditor(edit_strength=edit_strength)
        assert editor.edit_strength == edit_strength
```

#### 集成测试示例

```python
def test_full_pipeline():
    """测试完整的编辑管道"""
    editor = CausalEditor(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # 准备输入
    input_text = "Einstein was born in"
    editor.prepare_for_input(input_text)
    
    # 模拟激活
    batch_size, seq_len, hidden_dim = 1, 10, 2048
    activations = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 执行编辑
    edited = editor.edit_activations(
        activations=activations,
        generated_tokens=["Einstein", "was"],
        input_text=input_text
    )
    
    assert edited.shape == activations.shape
    assert not torch.equal(edited, activations)  # 应该有所改变
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_causal_editor.py

# 运行带覆盖率的测试
pytest --cov=causal_editor tests/

# 运行性能测试
pytest tests/performance/ --benchmark-only
```

## 📚 文档贡献

### 文档类型

1. **代码文档**: 内联注释和文档字符串
2. **API文档**: 自动生成的API参考
3. **用户指南**: 使用教程和示例
4. **开发文档**: 架构说明和设计决策

### 文档生成

```bash
# 生成API文档
sphinx-build -b html docs/ docs/_build/

# 本地预览
cd docs/_build/html
python -m http.server 8000
```

### 文档风格

- 使用清晰、简洁的语言
- 提供具体的代码示例
- 包含必要的图表和流程图
- 保持中英文一致性

## 🐛 问题报告

### 报告 Bug

请使用 GitHub Issues 报告 bug，包含以下信息：

```markdown
## Bug 描述
简要描述发生的问题

## 复现步骤
1. 步骤一
2. 步骤二
3. 观察到的问题

## 期望行为
描述您期望发生的行为

## 环境信息
- 操作系统: [e.g. Ubuntu 20.04]
- Python 版本: [e.g. 3.8.10]
- PyTorch 版本: [e.g. 2.0.1]
- CUDA 版本: [e.g. 11.8]
- GPU 型号: [e.g. RTX 3080]

## 额外信息
其他可能有用的信息，如错误日志、截图等
```

### Bug 优先级

- **严重**: 系统崩溃、数据丢失
- **高**: 核心功能无法使用
- **中**: 功能部分失效
- **低**: 小的UI问题、文档错误

## 💡 功能请求

### 提交新功能建议

```markdown
## 功能描述
描述您希望添加的功能

## 使用场景
说明这个功能在什么情况下有用

## 建议的实现方式
如果有想法，可以描述如何实现

## 替代方案
描述其他可能的解决方案

## 额外信息
其他相关信息
```

## 🚀 发布流程

### 版本号规范

我们使用 [语义化版本](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- `MAJOR`: 不兼容的API更改
- `MINOR`: 向后兼容的功能添加
- `PATCH`: 向后兼容的错误修复

### 发布检查清单

- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 版本号已更新
- [ ] CHANGELOG 已更新
- [ ] 性能回归测试通过

## 🏆 贡献者认可

我们感谢所有贡献者的努力！贡献类型包括：

- 💻 代码贡献
- 📖 文档改进
- 🐛 Bug 报告
- 💡 功能建议
- 🧪 测试用例
- 🎨 UI/UX 改进
- 🌐 本地化/翻译

## 📞 联系方式

如有疑问，请通过以下方式联系：

- GitHub Issues: 技术问题和 bug 报告
- GitHub Discussions: 一般讨论和问题
- Email: [维护者邮箱] (紧急情况)

## 📄 行为准则

我们致力于为每个人提供友好、安全和受欢迎的环境。请遵循以下原则：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

违反行为准则的行为将不被容忍。

---

再次感谢您对 CausalEditor 项目的贡献！您的参与使这个项目变得更好。🎉