import json
import logging
import os
import csv


import torch

# 设置详细的日志级别以查看形状调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from transformers import AutoTokenizer
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
os.environ['HF_HOME'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data'
os.environ['TRANSFORMERS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp" 
os.environ['HF_DATASETS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp"

# 设置transformers详细日志
# os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "causal_editor"))

# 导包
try:
    from causal_editor.core.causal_editor import CausalEditor
    from flashrag.config import Config
    from flashrag.utils import get_retriever
    # 使用Llama-2适配文件
    from modeling_llama_causal.modeling_llama_causal_Llama2 import CausalLlama2ForCausalLM
    # 导入RAG相关组件
    try:
        from causal_editor.dynamic.rag_config import RAGConfig
        from causal_editor.dynamic.fingerprint_builder import DynamicFingerprintBuilder
        from causal_editor.core.conflict_detector import CausalConflictDetector
        from causal_editor.core.counterfactual_editor import CounterfactualEditor
    except ImportError as e:
        logging.warning(f'导入失败: {e}')
        # TODO: 添加fallback逻辑
except ImportError as e:
    logging.error(
        f"导入组件失败。请确保您的 PYTHONPATH 设置正确。错误: {e}"
    )
    print("尝试调整 sys.path 以适应常见项目结构...")
    sys.path.insert(0, str(project_root.parent))
    sys.path.insert(0, str(project_root.parent / "causal_editor"))
    try:
        from causal_editor.core.causal_editor import CausalEditor
        from modeling_llama_causal.modeling_llama_causal_Llama2 import CausalLlama2ForCausalLM
        from causal_editor.dynamic.rag_config import RAGConfig
        from causal_editor.dynamic.fingerprint_builder import DynamicFingerprintBuilder
        from causal_editor.core.conflict_detector import CausalConflictDetector
        from causal_editor.core.counterfactual_editor import CounterfactualEditor
    except ImportError as e:
        logging.warning(f'导入失败: {e}')
        # TODO: 添加fallback逻辑

# --- 配置参数 ---
MODEL_NAME = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/llama2-7b-chat-hf"  # 本地模型路径
RESULT_DIR = "./result/llama2-7b-chat-hf_causal_editor"  # 结果保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 4096  # 输入长度
MAX_NEW_TOKENS = 50  # 生成的最大token数
TRUTHFULQA_DATA_PATH = "./TruthfulQA/TruthfulQA_sampled_38.csv"  # 使用采样后的数据集

# RAG系统配置
CAUSAL_EDITOR_CONFIG_PATH = "./configs/causal_editor.json"  # RAG配置文件路径
RAG_RETRIEVAL_CONFIG_PATH = "./configs/retrieval_config.yaml"  # RAG检索配置文件路径

USE_RAG_RETRIEVAL = True  # 启用RAG检索
RETRIEVAL_MODE = "rag_only"  # 检索模式：rag_only
ENABLE_DYNAMIC_THRESHOLD = True  # 启用动态阈值调整

# 确保结果目录存在
result_path = Path(RESULT_DIR)
result_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TruthfulQA Generation with Causal Editor")
print("=" * 80)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"检索模式: {RETRIEVAL_MODE}")
print(f"RAG配置: {CAUSAL_EDITOR_CONFIG_PATH}")
print(f"结果目录: {RESULT_DIR}")
print(f"TruthfulQA数据: {TRUTHFULQA_DATA_PATH}")
print("=" * 80)

def load_truthfulqa_data(data_path):
    """加载TruthfulQA数据集"""
    questions = []
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'type': row['Type'],
                'category': row['Category'],
                'question': row['Question'],
                'best_answer': row['Best Answer'],
                'correct_answers': row['Correct Answers'],
                'incorrect_answers': row['Incorrect Answers']
            })
    return questions

def main():
    # --- 步骤 1: 初始化RAG配置 ---
    print("\n--- 步骤 1: 初始化RAG配置 ---")
    
    config_path = Path(CAUSAL_EDITOR_CONFIG_PATH)
    causal_editor_config = RAGConfig(config_path=CAUSAL_EDITOR_CONFIG_PATH)  
    rag_config_dict = causal_editor_config.get_config()  
    dynamic_threshold_enabled = rag_config_dict.get("fallback_config", {}).get("enable_dynamic_threshold", False)

    config = Config(RAG_RETRIEVAL_CONFIG_PATH)
    
    # 初始化检索器
    rag_retriever = get_retriever(config)
    print("✅ RAG配置初始化完成")
    
    # --- 步骤 2: 加载集成RAG的Llama-2模型 ---
    print("\n--- 步骤 2: 加载集成RAG的Llama-2模型 ---")
    
    try:
        print("⏳ 正在加载模型...")
        
        model = CausalLlama2ForCausalLM.from_pretrained_with_dynamic_causal_editor(
            MODEL_NAME,
            device=DEVICE,
            rag_config=rag_config_dict,
        )
        print("✅ 模型加载成功")
        
        # 获取模型内置的tokenizer
        tokenizer = model.tokenizer
        if tokenizer is None:
            raise RuntimeError("模型未正确初始化tokenizer")
        print("✅ 分词器已从模型获取")
        
    except Exception as model_error:
        print(f"❌ 模型加载失败: {model_error}")
        sys.exit(1)
    
    model.eval()
    print(f"✅ 模型 {MODEL_NAME} 已加载到 {DEVICE}")
    
    # 验证 CausalEditor 和 RAG 集成
    causal_editor_instance = model.causal_editor
    
    if causal_editor_instance is None:
        raise RuntimeError("CausalEditor未正确初始化")
    
    print("✅ CausalEditor已自动初始化并附加到模型")
    
    # --- 步骤 3: 加载TruthfulQA数据集 ---
    print("\n--- 步骤 3: 加载TruthfulQA数据集 ---")
    
    try:
        questions = load_truthfulqa_data(TRUTHFULQA_DATA_PATH)
        print(f"✅ 成功加载 {len(questions)} 个TruthfulQA问题")
        
    except Exception as e:
        print(f"❌ 加载TruthfulQA数据失败: {e}")
        sys.exit(1)
    
    # --- 步骤 4: 执行生成测试 ---
    print("\n--- 步骤 4: 执行生成测试 ---")
    
    results = []
    start_time = datetime.now()
    
    for i, question_data in enumerate(questions):
        print(f"\n处理问题 {i + 1}/{len(questions)}: {question_data['category']}")
        
        user_query = question_data['question']
        
        try:
            # 构建 Llama-2 chat 格式的输入
            # messages = [
            #     {"role": "system", "content": "You are a helpful and accurate assistant. Please provide factual information."},
            #     {"role": "user", "content": user_query},
            # ]

            # # 应用 chat 模板
            # try:
            #     input_text = tokenizer.apply_chat_template(
            #         messages, tokenize=False, add_generation_prompt=True
            #     )
            # except Exception:
            #     # 回退到简单格式
            #     input_text = f"System: You are a helpful assistant.\nUser: {user_query}\nAssistant:"

            input_text = f"Q: {user_query}\nA:"
            
            # 编码输入
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,      # 单样本不需要padding
                truncation=True,    # 防止序列过长
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            if inputs['input_ids'].shape[0] > 1:
                print("维度错误-----------")
                print("维度错误-----------")
                print("维度错误-----------")
            # 验证输入长度并截断
            if inputs['input_ids'].shape[1] > MAX_LENGTH:
                inputs['input_ids'] = inputs['input_ids'][:, :MAX_LENGTH]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, :MAX_LENGTH]

            # 重置统计信息
            try:
                causal_editor_instance.reset_statistics()
            except Exception:
                pass

            # 准备输入 - 触发RAG检索
            try:
                if hasattr(causal_editor_instance, 'prepare_for_input'):
                    causal_editor_instance.prepare_for_input(user_query, rag_retriever)
            except Exception:
                pass
            
            # 生成回复
            generation_start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs
                )
            
            generation_time = time.time() - generation_start_time
            causal_editor_instance.finish_generation()
            
            # 解码生成的回复
            if outputs.shape[1] > inputs["input_ids"].shape[1]:
                response_text = tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            else:
                response_text = "[生成失败：无新token生成]"

            try: 
                # remove everything after 'Q:'
                response_text = response_text.split("Q:")[0].strip()
                # keep everything after A: 
                response_text = response_text.split("A:")[1].strip()
            except: 
                pass
            print("*"*50)
            print(response_text)
            print("*"*50)
            
            # 保存结果
            result_entry = {
                "question_id": i + 1,
                "type": question_data['type'],
                "category": question_data['category'],
                "question": user_query,
                "best_answer": question_data['best_answer'],
                "correct_answers": question_data['correct_answers'],
                "incorrect_answers": question_data['incorrect_answers'],
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
        "test_type": "truthfulqa_generation_causal_editor",
        "device": DEVICE,
        "retrieval_mode": RETRIEVAL_MODE,
        "rag_enabled": USE_RAG_RETRIEVAL,
        "dynamic_threshold_enabled": ENABLE_DYNAMIC_THRESHOLD,
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
        f.write("TruthfulQA Generation with Causal Editor 统计报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"🤖 模型: {MODEL_NAME}\n")
        f.write(f"💻 设备: {DEVICE}\n")
        f.write(f"🔍 检索模式: {RETRIEVAL_MODE}\n")
        f.write(f"📊 RAG启用: {'是' if USE_RAG_RETRIEVAL else '否'}\n")
        f.write(f"📈 动态阈值: {'启用' if ENABLE_DYNAMIC_THRESHOLD else '禁用'}\n")
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