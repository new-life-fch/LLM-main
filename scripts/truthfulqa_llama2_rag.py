import json
import logging
import os
import csv
from pathlib import Path
import sys
from datetime import datetime
import time
import traceback
from typing import Dict, List, Any, Optional

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


# 设置详细的日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置 Hugging Face 缓存目录
os.environ['HF_HOME'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data'
os.environ['TRANSFORMERS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp" 
os.environ['HF_DATASETS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp"

from flashrag.config import Config
from flashrag.utils import get_retriever

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- 配置参数 ---
MODEL_NAME = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/llama2-7b-chat-hf"  # 本地模型路径
RESULT_DIR = "./result/result_llama2_rag"  # 结果保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 4096  # 输入长度
MAX_NEW_TOKENS = 50  # 生成的最大token数
TRUTHFULQA_DATA_PATH = "./TruthfulQA/TruthfulQA_sampled.csv"  # 使用采样后的数据集

# 生成参数
TEMPERATURE = 0.0  # 温度设为0，确保确定性输出
DO_SAMPLE = False  # 不采样，使用贪婪解码

# RAG配置
RAG_CONFIG_PATH = "./configs/rag_only_config.json"  # RAG配置文件路径
RETRIEVAL_CONFIG_PATH = "./configs/retrieval_config.yaml"  # 检索配置文件路径
TOP_K_RETRIEVAL = 3  # 检索top-k个文档片段（与配置文件中rerank_topk保持一致）
MAX_CONTEXT_LENGTH = 512  # 每个检索片段的最大长度（与配置文件中chunk_size保持一致）

# 确保结果目录存在
result_path = Path(RESULT_DIR)
result_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TruthfulQA Generation with Llama2 + RAG")
print("=" * 80)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"结果目录: {RESULT_DIR}")
print(f"TruthfulQA数据: {TRUTHFULQA_DATA_PATH}")
print(f"最大新token数: {MAX_NEW_TOKENS}")
print(f"温度: {TEMPERATURE}")
print(f"采样: {DO_SAMPLE}")
print(f"检索Top-K: {TOP_K_RETRIEVAL}")
print(f"RAG配置文件: {RAG_CONFIG_PATH}")
print(f"检索配置文件: {RETRIEVAL_CONFIG_PATH}")
print("=" * 80)

def load_truthfulqa_data(data_path):
    """加载TruthfulQA数据集"""
    questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'type': row['Type'],
                'category': row['Category'],
                'question': row['Question'],
                'best_answer': row['Best Answer'],
                'correct_answers': row['Correct Answers'],
                'incorrect_answers': row['Incorrect Answers'],
            })
    return questions

def load_rag_config():
    """加载RAG配置文件"""
    try:
        # 加载主配置文件
        with open(RAG_CONFIG_PATH, 'r', encoding='utf-8') as f:
            rag_config = json.load(f)
        
        # 加载检索配置文件
        import yaml
        with open(RETRIEVAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            retrieval_config = yaml.safe_load(f)
        
        return rag_config, retrieval_config
    except Exception as e:
        print(f"⚠️ 配置文件加载失败: {e}")
        return None, None

def initialize_retriever():
    """初始化检索器"""
    try:
        # 加载配置文件
        rag_config, retrieval_config = load_rag_config()
        
        if rag_config is None or retrieval_config is None:
            print("⚠️ 配置文件加载失败，将使用模拟检索器")
            return None
        
        # 使用检索配置初始化FlashRAG配置
        config = Config(config_file_path=RETRIEVAL_CONFIG_PATH)
        
        # 初始化检索器
        retriever = get_retriever(config)
        print("✅ 检索器初始化成功")
        return retriever, rag_config
        
    except Exception as e:
        print(f"⚠️ 检索器初始化失败: {e}")
        print("将使用模拟检索器")
        return None, None

def retrieve_context(retriever, query):
    """检索相关上下文"""
    
    try:
        # 使用FlashRAG检索器
        results = retriever.search(query)
        
        # 提取文本内容
        contexts = []
        for result in results:
            if isinstance(result, dict):
                text = result.get('contents')
            else:
                text = str(result)
            
            contexts.append(text)
        
        return contexts
        
    except Exception as e:
        print(f"⚠️ 检索失败: {e}")
        # 返回模拟结果
        return [
            "This is a fallback retrieved context due to retrieval error.",
            "Another fallback context for demonstration."
        ]

def format_rag_prompt(question, contexts):
    """格式化RAG提示词"""
    context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""Based on the following retrieved information, please answer the question accurately and factually.

Retrieved Information:
{context_text}

Question: {question}

Please provide a factual and accurate answer based on the retrieved information above:"""
    
    return prompt

def main():
    # --- 步骤 1: 初始化检索器 ---
    print("\n--- 步骤 1: 初始化检索器 ---")
    
    retriever, rag_config = initialize_retriever()
    
    # --- 步骤 2: 加载原生Llama-2模型 ---
    print("\n--- 步骤 2: 加载原生Llama-2模型 ---")
    
    try:
        print("⏳ 正在加载模型...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ 模型加载成功")
        print("✅ 分词器加载成功")
        
    except Exception as model_error:
        print(f"❌ 模型加载失败: {model_error}")
        sys.exit(1)
    
    model.eval()
    print(f"✅ 原生模型 {MODEL_NAME} 已加载到 {DEVICE}")
    
    # --- 步骤 3: 加载TruthfulQA数据集 ---
    print("\n--- 步骤 3: 加载TruthfulQA采样数据集 ---")
    
    try:
        questions = load_truthfulqa_data(TRUTHFULQA_DATA_PATH)
        print(f"✅ 成功加载 {len(questions)} 个TruthfulQA问题")
        
    except Exception as e:
        print(f"❌ 加载TruthfulQA数据失败: {e}")
        sys.exit(1)
    
    # --- 步骤 4: 执行RAG生成测试 ---
    print("\n--- 步骤 4: 执行RAG生成测试 ---")
    
    results = []
    start_time = datetime.now()
    
    for i, question_data in enumerate(questions):
        print(f"\n处理问题 {i + 1}/{len(questions)}: {question_data['category']}")
        
        user_query = question_data['question']
        
        try:
            # 步骤 4.1: 检索相关上下文
            print(f"🔍 检索相关上下文...")
            # 生成回复
            generation_start_time = time.time()
            
            contexts = retrieve_context(retriever, user_query)
            print(f"✅ 检索到 {len(contexts)} 个相关片段")
            
            # 步骤 4.2: 构建RAG提示词
            rag_prompt = format_rag_prompt(user_query, contexts)
            
            # 构建 Llama-2 chat 格式的输入
            messages = [
                {"role": "system", "content": "You are a helpful and accurate assistant. Please provide factual information."},
                {"role": "user", "content": rag_prompt},
            ]

            # 应用 chat 模板
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # 回退到简单格式
                input_text = f"System: You are a helpful assistant.\nUser: {rag_prompt}\nAssistant:"
            
            # 编码输入
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,      # 单样本不需要padding
                truncation=True,    # 防止序列过长
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            # 验证输入长度并截断
            if inputs['input_ids'].shape[1] > MAX_LENGTH:
                inputs['input_ids'] = inputs['input_ids'][:, :MAX_LENGTH]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, :MAX_LENGTH]
            
            
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    num_beams=1,
                    early_stopping=True,
                )
            
            generation_time = time.time() - generation_start_time
            
            # 解码生成的回复
            if outputs.shape[1] > inputs["input_ids"].shape[1]:
                response_text = tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            else:
                response_text = "[生成失败：无新token生成]"
            
            # 保存结果
            result_entry = {
                "question_id": i + 1,
                "type": question_data['type'],
                "category": question_data['category'],
                "question": user_query,
                "best_answer": question_data['best_answer'],
                "correct_answers": question_data['correct_answers'],
                "incorrect_answers": question_data['incorrect_answers'],
                "retrieved_contexts": contexts,
                "rag_prompt": rag_prompt,
                "generated_answer": response_text,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            results.append(result_entry)
            
            print(f"✅ 问题 {i + 1} 处理完成")
            
        except Exception as e:
            print(f"❌ 问题 {i + 1} 处理失败: {e}")
            
            # 记录错误结果
            result_entry = {
                "question_id": i + 1,
                "type": question_data['type'],
                "category": question_data['category'],
                "question": user_query,
                "best_answer": question_data['best_answer'],
                "correct_answers": question_data['correct_answers'],
                "incorrect_answers": question_data['incorrect_answers'],
                "generated_answer": f"ERROR: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            results.append(result_entry)
            continue
    
    end_time = datetime.now()
    
    # --- 步骤 5: 保存结果 ---
    print("\n--- 步骤 5: 保存结果 ---")
    
    # 计算统计信息
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    total_generation_time = sum(r.get('generation_time', 0) for r in successful_tests)
    avg_generation_time = total_generation_time / len(successful_tests) if successful_tests else 0
    
    # 保存干净的结果文件（仅包含问题和答案）
    clean_results_path = result_path / "truthfulqa_clean_results.json"
    
    clean_results = []
    for result in results:
        if result.get('success', False):
            clean_entry = {
                "question_id": result['question_id'],
                "question": result['question'],
                "generated_answer": result['generated_answer'],
                "category": result['category'],
                "type": result['type']
            }
            clean_results.append(clean_entry)
    
    with open(clean_results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 干净结果文件已保存: {clean_results_path}")
    
    # 保存符合GPT-3微调评估格式的结果
    gpt3_format_path = result_path / "truthfulqa_judge_format.jsonl"
    
    with open(gpt3_format_path, "w", encoding="utf-8") as f:
        for result in results:
            if result.get('success', False):
                gpt3_entry = {
                    "question": result['question'],
                    "answer": result['generated_answer'],
                    "category": result['category'],
                    "type": result['type']
                }
                f.write(json.dumps(gpt3_entry, ensure_ascii=False) + "\n")
    
    print(f"✅ GPT-3格式结果已保存: {gpt3_format_path}")
    
    # 保存包含统计信息和调试信息的详细文件
    debug_results_path = result_path / "truthfulqa_debug_results.json"
    
    debug_summary = {
        "model_name": MODEL_NAME,
        "test_type": "truthfulqa_generation_llama2_rag",
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "rag_config_path": RAG_CONFIG_PATH,
        "retrieval_config_path": RETRIEVAL_CONFIG_PATH,
        "test_start_time": start_time.isoformat(),
        "test_end_time": end_time.isoformat(),
        "total_test_time": str(end_time - start_time),
        "total_questions": len(questions),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "success_rate": len(successful_tests) / len(questions) * 100 if questions else 0,
        "avg_generation_time": avg_generation_time,
        "detailed_results": results
    }

    with open(debug_results_path, "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 调试信息文件已保存: {debug_results_path}")
    
    # 保存统计报告
    stats_report_path = result_path / "truthfulqa_statistics_report.txt"
    
    with open(stats_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TruthfulQA Generation with Llama2 + RAG 统计报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"🤖 模型: {MODEL_NAME}\n")
        f.write(f"💻 设备: {DEVICE}\n")
        f.write(f"🎯 最大新token数: {MAX_NEW_TOKENS}\n")
        f.write(f"🌡️ 温度: {TEMPERATURE}\n")
        f.write(f"🎲 采样: {'是' if DO_SAMPLE else '否'}\n")
        f.write(f"🔍 检索Top-K: {TOP_K_RETRIEVAL}\n")
        f.write(f"📏 最大上下文长度: {MAX_CONTEXT_LENGTH}\n")
        f.write(f"📋 RAG配置文件: {RAG_CONFIG_PATH}\n")
        f.write(f"📋 检索配置文件: {RETRIEVAL_CONFIG_PATH}\n")
        f.write(f"⏰ 测试时间: {start_time} - {end_time}\n")
        f.write(f"⏱️ 测试耗时: {end_time - start_time}\n")
        f.write(f"📊 问题总数: {len(questions)}\n")
        f.write(f"✅ 成功数量: {len(successful_tests)}\n")
        f.write(f"❌ 失败数量: {len(failed_tests)}\n")
        f.write(f"✅ 成功率: {len(successful_tests) / len(questions) * 100:.1f}%\n")
        f.write(f"⚡ 平均生成时间: {avg_generation_time:.3f}s\n")
        f.write(f"📁 结果文件:\n")
        f.write(f"  - 干净结果: {clean_results_path}\n")
        f.write(f"  - GPT-3格式: {gpt3_format_path}\n")
        f.write(f"  - 调试信息: {debug_results_path}\n")
    
    print(f"✅ 统计报告已保存: {stats_report_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print(f"总问题数: {len(questions)}")
    print(f"成功生成: {len(successful_tests)}")
    print(f"失败数量: {len(failed_tests)}")
    print(f"成功率: {len(successful_tests) / len(questions) * 100:.1f}%")
    print(f"平均生成时间: {avg_generation_time:.3f}s")
    print("\n📁 生成的文件:")
    print(f"  - 干净结果: truthfulqa_clean_results.json")
    print(f"  - GPT-3格式: truthfulqa_judge_format.jsonl")
    print(f"  - 调试信息: truthfulqa_debug_results.json")
    print(f"  - 统计报告: truthfulqa_statistics_report.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()