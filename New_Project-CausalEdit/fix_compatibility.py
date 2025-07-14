#!/usr/bin/env python3
"""
修复CausalEditor版本兼容性问题的脚本
主要解决transformers和torchvision的版本冲突
"""

import subprocess
import sys
import logging

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"执行: {cmd}")
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return False
        else:
            print(f"成功: {result.stdout}")
            return True
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def fix_compatibility():
    """修复版本兼容性"""
    print("🔧 开始修复CausalEditor版本兼容性问题...")
    
    # 方案1: 降级transformers版本
    print("\n📦 方案1: 降级transformers版本")
    commands = [
        "pip uninstall transformers torchvision -y",
        "pip install transformers==4.30.0",
        "conda install torchvision==0.15.0 -c pytorch -y"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print("⚠️ 方案1失败，尝试方案2...")
            break
    else:
        print("✅ 方案1成功！")
        return True
    
    # 方案2: 使用conda管理所有包
    print("\n📦 方案2: 使用conda重新安装")
    conda_commands = [
        "conda uninstall transformers torchvision pytorch -y",
        "conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y",
        "conda install transformers==4.30.0 -c conda-forge -y"
    ]
    
    for cmd in conda_commands:
        if not run_command(cmd):
            print("⚠️ 方案2失败...")
            break
    else:
        print("✅ 方案2成功！")
        return True
    
    # 方案3: 创建新环境
    print("\n📦 方案3: 建议创建新的conda环境")
    print("请手动执行以下命令：")
    print("conda create -n causal-editor python=3.10 -y")
    print("conda activate causal-editor") 
    print("conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y")
    print("conda install transformers==4.30.0 faiss-cpu -c conda-forge -y")
    print("pip install -r requirements.txt")
    
    return False

def test_imports():
    """测试导入是否成功"""
    print("\n🧪 测试导入...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        # 测试有问题的导入
        from transformers.models.llama.modeling_llama import LlamaModel
        print("✅ Llama模型导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        return False

if __name__ == "__main__":
    print("CausalEditor 版本兼容性修复工具")
    print("=" * 50)
    
    # 检查当前状态
    print("🔍 检查当前版本...")
    run_command("python -c \"import torch; print('PyTorch:', torch.__version__)\"")
    run_command("python -c \"import transformers; print('Transformers:', transformers.__version__)\"")
    
    # 尝试修复
    if fix_compatibility():
        print("\n🎉 兼容性问题已修复！")
        if test_imports():
            print("✅ 所有测试通过，可以开始使用CausalEditor")
        else:
            print("⚠️ 仍有问题，建议使用方案3创建新环境")
    else:
        print("\n⚠️ 自动修复失败，请按照方案3手动创建新环境") 