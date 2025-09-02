import os
from huggingface_hub import snapshot_download, login

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 尝试登录 Hugging Face（如果有token的话）
try:
    # 如果设置了HF_TOKEN环境变量，则使用它登录
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("已使用HF_TOKEN登录")
except Exception as e:
    print(f"登录失败，将尝试匿名下载: {e}")

# 模型名称
repo_id = "meta-llama/Llama-2-7b-chat-hf"

# 您指定的本地下载路径
local_dir = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/llama2-7b-chat-hf"

print(f"正在从 HF-Mirror 镜像站下载模型 '{repo_id}' 到 '{local_dir}'...")

try:
    # 确保目标目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    # 使用 snapshot_download 函数下载模型
    # local_dir_use_symlinks=False 参数可以避免在某些文件系统上出现问题
    # resume_download=True 允许断点续传
    snapshot_download(
        repo_id=repo_id, 
        local_dir=local_dir, 
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.bin"]  # 只下载safetensors格式，忽略bin文件
    )
    print("模型下载成功！")
except Exception as e:
    print(f"模型下载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 设置HF_TOKEN环境变量（如果模型需要认证）")
    print("3. 尝试使用官方Hub: unset HF_ENDPOINT")
    print("4. 检查模型名称是否正确")