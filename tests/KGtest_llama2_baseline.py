#!/usr/bin/env python3
"""
Llama-2-7b-chat 基线测试（无CausalEditor）
用于与集成CausalEditor的版本进行对比

测试目的：
1. 获取原生模型的自然回复作为基线
2. 与CausalEditor修改后的回复进行对比分析
3. 评估CausalEditor对模型输出的实际影响

使用方法:
1. 确保有足够的GPU内存 (推荐16GB+)
2. 安装所需依赖
3. 运行此脚本获取基线结果
"""

import os
import json
import logging

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pathlib import Path
    import sys
    from datetime import datetime
    import time
    import traceback
    from typing import Dict, List, Any, Optional
except ImportError as e:
    logging.warning(f'导入失败: {e}')
    # TODO: 添加fallback逻辑

# 设置 Hugging Face 缓存目录
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/huggingface/datasets'

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Llama-2 基线配置 ---
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # 原始在线模型名称
MODEL_NAME = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"  # 本地模型路径
RESULT_DIR = "./result_llama2_baseline"  # 基线结果保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
MAX_LENGTH = 4096
DEBUG_MODE = True
SAVE_INTERMEDIATE_RESULTS = True

# 确保结果目录存在
result_path = Path(RESULT_DIR)
result_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Llama-2-7b-chat 基线测试（无CausalEditor）")
print("=" * 80)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"结果目录: {RESULT_DIR}")
print("=" * 80)

# 检查GPU内存
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU内存: {gpu_memory:.1f} GB")
    if gpu_memory < 15:
        print("⚠️  警告: GPU内存可能不足，推荐至少16GB用于Llama-2-7b模型")
        print("   建议启用CPU offloading或使用更小的模型")
else:
    print("⚠️  警告: 未检测到CUDA，将使用CPU运行（速度会很慢）")

# --- 步骤 1: 加载原生 Llama-2 模型（无CausalEditor） ---
print("\n--- 步骤 1: 加载原生 Llama-2 模型（无CausalEditor） ---")

try:
    # 加载分词器
    logging.info("正在加载 Llama-2 分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        local_files_only=True  # 强制使用本地缓存，避免网络请求
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("设置 pad_token 为 eos_token")

    # 加载原生模型（使用与CausalEditor版本相同的参数以确保公平比较）
    logging.info("正在加载原生 Llama-2 模型...")
    print("⏳ 正在下载和加载模型，这可能需要几分钟...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # 与CausalEditor版本保持一致
        device_map="auto",  # 自动处理设备分配，无需手动to()
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        max_memory={0: "19GB"},  # 针对24GB显存优化，保留2GB系统缓冲
        local_files_only=True  # 强制使用本地缓存，避免网络请求
    )
    
    model.eval()
    print(f"✅ 原生模型 {MODEL_NAME} 已加载到 {DEVICE}")
    
    # 显示模型信息
    if DEBUG_MODE:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📈 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型设备: {next(model.parameters()).device}")
        print(f"  模型类型: {type(model).__name__}")
    
    print("✅ 原生Llama-2模型加载完成（无CausalEditor干预）")

except Exception as e:
    logging.error(f"❌ 加载原生模型失败: {e}")
    logging.error(f"错误详情: {traceback.format_exc()}")
    print("\n💡 故障排除建议:")
    print("1. 检查GPU内存是否足够 (推荐16GB+)")
    print("2. 确保网络连接正常以下载模型")
    print("3. 验证 Hugging Face 访问权限")
    print("4. 考虑使用更小的模型进行测试")
    sys.exit(1)

# --- 步骤 2: 执行基线推理测试（与CausalEditor版本使用相同测试用例） ---
print("\n--- 步骤 2: 执行基线推理测试 ---")

# 使用与test_llama2_example.py完全相同的测试用例
test_cases = [
    # {
    #     "category": "历史知识测试",
    #     "prompt": "The American Civil War ended in 1975. Is this correct?",
    #     "expected": "No, the American Civil War ended in 1865",
    #     "difficulty": "medium"
    # },
    {
        "category": "科学常识测试", 
        "prompt": "Water boils at 50°C at sea level. Explain why.",
        "expected": "This is incorrect. Water boils at 100°C at sea level",
        "difficulty": "easy"
    }
    # {
    #     "category": "数学逻辑测试",
    #     "prompt": "If 2+2=5, what is 3+3?",
    #     "expected": "The premise is incorrect. 2+2=4, not 5",
    #     "difficulty": "hard"
    # }
]

print(f"准备了 {len(test_cases)} 个测试用例（与CausalEditor版本保持一致）")

results = []
test_start_time = datetime.now()

for i, test_case in enumerate(test_cases):
    case_start_time = time.time()
    user_query = test_case["prompt"]
    expected_answer = test_case["expected"]
    category = test_case["category"]
    difficulty = test_case["difficulty"]
    
    print(f"\n🧪 基线测试 {i + 1}/{len(test_cases)}: {category} ({difficulty})")
    print(f"❓ 问题: {user_query}")
    print(f"🎯 期望: {expected_answer}")
    
    try:
        # 构建 Llama-2 chat 格式的输入（与CausalEditor版本保持一致）
        messages = [
            {"role": "system", "content": "You are a helpful and accurate assistant. Please provide factual information."},
            {"role": "user", "content": user_query},
        ]

        # 应用 chat 模板
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(DEVICE)
        
        print(f"📝 输入token数量: {inputs['input_ids'].shape[1]}")

        # 生成回复（使用与CausalEditor版本相同的参数）
        generation_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH,  # 限制生成长度
                num_beams=1,
                do_sample=False,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        
        generation_time = time.time() - generation_start_time
        
        # 解码回复
        response_text = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        
        print(f"🤖 基线回复: {response_text}")
        print(f"⏱️  生成耗时: {generation_time:.3f}s")
        
        # 记录结果
        case_end_time = time.time()
        result_entry = {
            "question_id": i + 1,
            "category": category,
            "difficulty": difficulty,
            "question": user_query,
            "expected_answer": expected_answer,
            "generated_answer": response_text,
            "generation_time": generation_time,
            "total_case_time": case_end_time - case_start_time,
            "input_tokens": inputs['input_ids'].shape[1],
            "output_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
            "model_type": "baseline_llama2",
            "causal_editor_used": False,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result_entry)
        
        print("✅ 基线测试完成")
        
    except Exception as e:
        logging.error(f"❌ 基线测试用例 {i+1} 执行失败: {e}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        
        # 记录错误结果
        result_entry = {
            "question_id": i + 1,
            "category": category,
            "difficulty": difficulty,
            "question": user_query,
            "expected_answer": expected_answer,
            "generated_answer": f"ERROR: {str(e)}",
            "error": str(e),
            "model_type": "baseline_llama2",
            "causal_editor_used": False,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result_entry)
        continue

test_end_time = datetime.now()

# --- 步骤 3: 保存基线结果 ---
print(f"\n--- 步骤 3: 保存基线测试结果 ---")

# 计算统计信息
successful_tests = [r for r in results if 'error' not in r]
failed_tests = [r for r in results if 'error' in r]
total_generation_time = sum(r.get('generation_time', 0) for r in successful_tests)
avg_generation_time = total_generation_time / len(successful_tests) if successful_tests else 0

print(f"\n📊 基线测试统计摘要:")
print(f"  ✅ 成功测试: {len(successful_tests)} / {len(results)}")
print(f"  ❌ 失败测试: {len(failed_tests)}")
print(f"  ⏱️  平均生成时间: {avg_generation_time:.3f}s")
print(f"  🕐 总测试时间: {test_end_time - test_start_time}")

# 保存详细结果
results_json_path = result_path / "llama2_baseline_results.json"
test_summary = {
    "model_name": MODEL_NAME,
    "test_type": "llama2_baseline_test",
    "model_type": "baseline_llama2",
    "causal_editor_used": False,
    "device": DEVICE,
    "test_start_time": test_start_time.isoformat(),
    "test_end_time": test_end_time.isoformat(),
    "total_test_time": str(test_end_time - test_start_time),
    "total_questions": len(test_cases),
    "successful_tests": len(successful_tests),
    "failed_tests": len(failed_tests),
    "success_rate": len(successful_tests) / len(test_cases) * 100,
    "avg_generation_time": avg_generation_time,
    "results": results,
}

with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(test_summary, f, indent=2, ensure_ascii=False)

# 保存简化报告
report_path = result_path / "llama2_baseline_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("Llama-2-7b-chat 基线测试报告（无CausalEditor）\n")
    f.write("=" * 80 + "\n")
    f.write(f"🤖 模型: {MODEL_NAME}\n")
    f.write(f"📋 模型类型: 原生Llama-2（无CausalEditor干预）\n")
    f.write(f"💻 设备: {DEVICE}\n")
    f.write(f"⏰ 测试时间: {test_start_time} - {test_end_time}\n")
    f.write(f"⏱️ 测试耗时: {test_end_time - test_start_time}\n")
    f.write(f"📊 问题总数: {len(test_cases)}\n")
    f.write(f"✅ 成功率: {len(successful_tests) / len(test_cases) * 100:.1f}%\n")
    f.write(f"⚡ 平均生成时间: {avg_generation_time:.3f}s\n\n")
    
    f.write("📝 注意: 此为基线测试，未使用CausalEditor\n")
    f.write("🔗 可与 result_llama2/llama2_test_report.txt 对比分析CausalEditor效果\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("详细基线测试结果\n")
    f.write("=" * 80 + "\n")
    
    for i, result in enumerate(results):
        status = "✅" if 'error' not in result else "❌"
        f.write(f"\n{status} 基线测试 {i + 1}: [{result.get('category', 'Unknown')}]\n")
        f.write(f"❓ 问题: {result['question']}\n")
        f.write(f"🎯 期望: {result['expected_answer']}\n")
        f.write(f"🤖 基线回答: {result['generated_answer']}\n")
        if 'error' not in result:
            f.write(f"⏱️ 生成时间: {result.get('generation_time', 0):.3f}s\n")
        f.write("-" * 60 + "\n")

print(f"\n💾 基线测试结果已保存:")
print(f"  📄 详细JSON: {results_json_path}")
print(f"  📋 可读报告: {report_path}")

print("\n" + "="*80)
print("🎉 Llama-2-7b-chat 基线测试完成！")
print("📊 已获取原生模型的自然回复作为对比基线")
print("🔗 建议接下来运行 test_llama2_example.py 进行CausalEditor版本测试")
print("📈 然后对比两个结果目录分析CausalEditor的实际效果:")
print(f"   - 基线结果: {RESULT_DIR}/")
print(f"   - CausalEditor结果: ./result_llama2/")
print("="*80)