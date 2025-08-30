# test.py - 动态因果编辑测试脚本（增强调试版本）

import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import pandas as pd
import sys
from datetime import datetime
import time
import traceback
from typing import Dict, List, Any, Optional
import numpy as np

# 将项目根目录添加到sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "causal_editor"))

try:
    from causal_editor.core.causal_editor import CausalEditor
    from modeling_llama_causal.modeling_llama_causal import CausalLlamaForCausalLM
except ImportError as e:
    logging.error(
        f"导入 CausalEditor 组件失败。请确保您的 PYTHONPATH 设置正确。错误: {e}"
    )
    print("尝试调整 sys.path 以适应常见项目结构...")
    sys.path.insert(0, str(project_root.parent))
    sys.path.insert(0, str(project_root.parent / "causal_editor"))
    from causal_editor.core.causal_editor import CausalEditor
    from modeling_llama_causal.modeling_llama_causal import CausalLlamaForCausalLM

# 设置日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 调试工具类 ---
class DebugMonitor:
    """调试监控器，用于跟踪数据流和性能"""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.conflict_history = []
        self.edit_history = []
        self.layer_activations = {}
        
    def start_step(self, step_name: str):
        """开始一个调试步骤"""
        self.current_step = step_name
        self.step_start_time = time.time()
        logging.debug(f"🔍 开始调试步骤: {step_name}")
        
    def end_step(self):
        """结束当前调试步骤"""
        if hasattr(self, 'current_step'):
            duration = time.time() - self.step_start_time
            self.step_times.append({
                'step': self.current_step,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            logging.debug(f"✅ 完成调试步骤: {self.current_step} (耗时: {duration:.3f}s)")
            
    def record_conflict(self, layer_id: str, conflict_info: Dict[str, Any]):
        """记录冲突检测信息"""
        self.conflict_history.append({
            'layer_id': layer_id,
            'has_conflict': conflict_info.get('has_conflict', False),
            'confidence': conflict_info.get('confidence', 0.0),
            'timestamp': datetime.now().isoformat()
        })
        
    def record_edit(self, layer_id: str, edit_info: Dict[str, Any]):
        """记录编辑操作信息"""
        self.edit_history.append({
            'layer_id': layer_id,
            'edit_applied': True,
            'edit_info': edit_info,
            'timestamp': datetime.now().isoformat()
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """获取调试摘要"""
        total_conflicts = sum(1 for c in self.conflict_history if c['has_conflict'])
        total_edits = len(self.edit_history)
        avg_confidence = np.mean([c['confidence'] for c in self.conflict_history]) if self.conflict_history else 0.0
        
        return {
            'total_steps': len(self.step_times),
            'total_time': sum(s['duration'] for s in self.step_times),
            'total_conflicts': total_conflicts,
            'total_edits': total_edits,
            'avg_confidence': avg_confidence,
            'step_details': self.step_times,
            'conflict_details': self.conflict_history,
            'edit_details': self.edit_history
        }

# --- 配置 ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULT_DIR = "./result"  # 结果保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # 调试时保持批处理大小较小
MAX_LENGTH = 500  # 生成的最大 token 数
DEBUG_MODE = True  # 启用详细调试模式
SAVE_INTERMEDIATE_RESULTS = True  # 保存中间结果

# 确保结果目录存在
result_path = Path(RESULT_DIR)
result_path.mkdir(parents=True, exist_ok=True)

# 初始化调试监控器
debug_monitor = DebugMonitor()

# --- 步骤 1: 加载集成 CausalEditor 的模型（动态模式）---
print("--- 步骤 1: 加载集成 CausalEditor 的模型（动态模式）---")
debug_monitor.start_step("模型加载")

try:
    # 加载分词器
    logging.debug("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确保 pad_token 已设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.debug("设置 pad_token 为 eos_token")

    # 加载模型
    logging.debug("正在加载模型...")
    model = CausalLlamaForCausalLM.from_pretrained_with_dynamic_causal_editor(
        MODEL_NAME,
        torch_dtype=torch.float16,  # 使用float16减少内存使用
        device_map="auto",  # 自动分发到可用设备
        low_cpu_mem_usage=True,  # 减少CPU内存使用
        trust_remote_code=True,  # 信任远程代码
        edit_strength=1.5,  # 编辑强度
        num_middle_layers=5,  # 参与编辑的层数
        similarity_threshold=0.6,  # 相似度阈值
        conflict_threshold=0.9,  # 冲突检测阈值
        device=DEVICE,  # 设备
    )
    model.eval()
    model.to(DEVICE)
    print(f"✅ 模型 {MODEL_NAME} 已加载到 {DEVICE}")
    
    # 打印模型信息
    if DEBUG_MODE:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.debug(f"模型参数总数: {total_params:,}")
        logging.debug(f"可训练参数: {trainable_params:,}")
        logging.debug(f"模型设备: {next(model.parameters()).device}")

    debug_monitor.end_step()
    debug_monitor.start_step("CausalEditor验证")
    
    # 获取已经初始化的CausalEditor实例（from_pretrained_with_dynamic_causal_editor已经自动初始化）
    causal_editor_instance = model.causal_editor
    
    if causal_editor_instance is None:
        raise RuntimeError("CausalEditor未正确初始化，请检查from_pretrained_with_dynamic_causal_editor方法")
    
    print("✅ CausalEditor（动态模式）已自动初始化并附加到模型")
    print("✅ Tokenizer 已自动设置到模型")
    
    # 打印CausalEditor配置
    if DEBUG_MODE:
        editor_stats = causal_editor_instance.get_statistics()
        logging.debug(f"CausalEditor配置: {json.dumps(editor_stats, indent=2, ensure_ascii=False)}")
    
    debug_monitor.end_step()

except Exception as e:
    logging.error(f"❌ 加载模型或初始化 CausalEditor 失败: {e}")
    logging.error(f"错误详情: {traceback.format_exc()}")
    sys.exit(1)

# --- 步骤 2: 执行推理并观察数据流 ---
print("--- 步骤 2: 执行推理并观察数据流 ---")
debug_monitor.start_step("测试用例准备")


test_cases = [
    {
        "category": "历史幻觉测试",
        "prompt": "The American Civil War ended in 1975 when President Lincoln signed the peace treaty.",
        "expected": "The Civil War ended in 1865, and Lincoln was assassinated that year",
        "difficulty": "hard"
    },
    
    # 科学事实错误 - 容易产生幻觉
    {
        "category": "科学幻觉测试",
        "prompt": "Water boils at 50 degrees Celsius at sea level. Explain why this happens.",
        "expected": "Water boils at 100°C at sea level, not 50°C",
        "difficulty": "medium"
    }
    
]

print(f"准备了 {len(test_cases)} 个测试用例，涵盖 {len(set(tc['category'] for tc in test_cases))} 个类别")
debug_monitor.end_step()

# 记录测试结果
causal_results = []
test_start_time = datetime.now()
debug_monitor.start_step("推理测试执行")

# 为每个测试用例执行推理
for i, test_case in enumerate(test_cases):
    case_start_time = time.time()
    user_query = test_case["prompt"]
    expected_answer = test_case["expected"]
    category = test_case["category"]
    difficulty = test_case["difficulty"]
    
    # 简化测试用例信息输出
    print(f"\n测试用例 {i + 1}/{len(test_cases)}: {category} ({difficulty})")
    print(f"问题: {user_query}")
    print(f"期望答案: {expected_answer}")
    
    debug_monitor.start_step(f"测试用例_{i+1}_处理")
    
    try:
        # 准备消息
        messages = [
            {"role": "system", "content": "You are a helpful and accurate assistant."},
            {"role": "user", "content": user_query},
        ]

        # 应用聊天模板
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 分词化
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(DEVICE)
        
        if DEBUG_MODE:
            print(f"输入token数量: {inputs['input_ids'].shape[1]}")

        # 重置CausalEditor统计信息（为每个测试用例单独统计）
        causal_editor_instance.reset_statistics()

        # 使用新方法为CausalEditor准备输入
        causal_editor_instance.prepare_for_input(user_query)
        
        # 设置生成上下文信息（重要：这样CausalEditor才能正确工作）
        model.model.set_generation_context(
            generated_tokens=[],  # 初始为空
            context_tokens=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            input_text=input_text.strip()
        )
        
        # 执行生成（动态模式下会实时进行冲突检测和编辑）
        generation_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 减少生成长度，更容易观察冲突
                num_beams=1,  # 调试时保持简单
                do_sample=False,  # 启用采样以增加多样性
                top_k=10,  # 减少top_k，增加冲突可能性
                top_p=0.8,  # 降低top_p
                temperature=1.0,  # 增加温度，增加随机性
                repetition_penalty=1.1,  # 降低重复惩罚
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generation_time = time.time() - generation_start_time
        
        # 解码生成的文本
        response_text = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        
        # 获取本次推理的CausalEditor统计信息
        current_stats = causal_editor_instance.get_statistics()
        
        print(f"生成回复: {response_text}")
        print(f"生成耗时: {generation_time:.3f}s")
        
        if DEBUG_MODE:
            conflict_stats = current_stats.get('conflict_detector_stats', {})
            edit_stats = current_stats.get('counterfactual_editor_stats', {})
            print(f"冲突检测: {conflict_stats.get('detection_count', 0)}, 发现冲突: {conflict_stats.get('conflict_count', 0)}")
            print(f"执行编辑: {edit_stats.get('edit_count', 0)}, 成功编辑: {edit_stats.get('successful_edits', 0)}")
        
        # 记录详细结果
        case_end_time = time.time()
        result_entry = {
            "question_id": i + 1,
            "category": category,
            "difficulty": difficulty,
            "question": user_query,
            "expected_answer": expected_answer,
            "generated_answer": response_text,
            "input_text": input_text.strip(),
            "generation_time": generation_time,
            "total_case_time": case_end_time - case_start_time,
            "input_tokens": inputs['input_ids'].shape[1],
            "output_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
            "causal_editor_stats": current_stats,
            "timestamp": datetime.now().isoformat(),
        }
        causal_results.append(result_entry)
        
        # 保存中间结果（如果启用）
        if SAVE_INTERMEDIATE_RESULTS and (i + 1) % 5 == 0:
            intermediate_path = result_path / f"intermediate_results_{i+1}.json"
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump({
                    "completed_cases": i + 1,
                    "total_cases": len(test_cases),
                    "results": causal_results,
                    "debug_summary": debug_monitor.get_summary()
                }, f, indent=2, ensure_ascii=False)
            if DEBUG_MODE:
                print(f"已保存中间结果到: {intermediate_path}")
        
        debug_monitor.end_step()
        
    except Exception as e:
        logging.error(f"❌ 测试用例 {i+1} 执行失败: {e}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        
        # 记录失败的测试用例
        result_entry = {
            "question_id": i + 1,
            "category": category,
            "difficulty": difficulty,
            "question": user_query,
            "expected_answer": expected_answer,
            "generated_answer": f"ERROR: {str(e)}",
            "input_text": "",
            "generation_time": 0,
            "total_case_time": time.time() - case_start_time,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        causal_results.append(result_entry)
        debug_monitor.end_step()
        continue

test_end_time = datetime.now()
debug_monitor.end_step()

print(f"\n所有测试用例执行完成！")
print(f"总耗时: {test_end_time - test_start_time}")
print(f"成功执行: {len([r for r in causal_results if 'error' not in r])} / {len(test_cases)}")

# --- 步骤 3: 保存动态CausalEditor测试结果 ---
print("\n--- 步骤 3: 保存动态CausalEditor测试结果 ---")
debug_monitor.start_step("结果分析和保存")

# 获取最终的 CausalEditor 统计信息
final_stats = causal_editor_instance.get_statistics()
debug_summary = debug_monitor.get_summary()

# 计算测试统计信息
successful_tests = [r for r in causal_results if 'error' not in r]
failed_tests = [r for r in causal_results if 'error' in r]
total_generation_time = sum(r.get('generation_time', 0) for r in successful_tests)
avg_generation_time = total_generation_time / len(successful_tests) if successful_tests else 0

# 按类别统计
category_stats = {}
for result in causal_results:
    category = result.get('category', 'Unknown')
    if category not in category_stats:
        category_stats[category] = {'total': 0, 'success': 0, 'avg_time': 0}
    category_stats[category]['total'] += 1
    if 'error' not in result:
        category_stats[category]['success'] += 1
        category_stats[category]['avg_time'] += result.get('generation_time', 0)

# 计算平均时间
for category in category_stats:
    if category_stats[category]['success'] > 0:
        category_stats[category]['avg_time'] /= category_stats[category]['success']

# 按难度统计
difficulty_stats = {}
for result in causal_results:
    difficulty = result.get('difficulty', 'Unknown')
    if difficulty not in difficulty_stats:
        difficulty_stats[difficulty] = {'total': 0, 'success': 0, 'avg_time': 0}
    difficulty_stats[difficulty]['total'] += 1
    if 'error' not in result:
        difficulty_stats[difficulty]['success'] += 1
        difficulty_stats[difficulty]['avg_time'] += result.get('generation_time', 0)

# 计算平均时间
for difficulty in difficulty_stats:
    if difficulty_stats[difficulty]['success'] > 0:
        difficulty_stats[difficulty]['avg_time'] /= difficulty_stats[difficulty]['success']

print(f"\n测试统计摘要:")
print(f"  成功测试: {len(successful_tests)} / {len(causal_results)}")
print(f"  失败测试: {len(failed_tests)}")
print(f"  平均生成时间: {avg_generation_time:.3f}s")
print(f"  总调试步骤: {debug_summary['total_steps']}")
print(f"  总冲突检测: {debug_summary['total_conflicts']}")
print(f"  总编辑操作: {debug_summary['total_edits']}")

print(f"\n动态CausalEditor 总统计信息:")
print(json.dumps(final_stats, indent=2, ensure_ascii=False))

# 保存详细结果到JSON
causal_results_json_path = result_path / "enhanced_dynamic_causal_editor_results.json"
causal_test_summary = {
    "model_name": MODEL_NAME,
    "test_type": "enhanced_dynamic_causal_editor",
    "device": DEVICE,
    "debug_mode": DEBUG_MODE,
    "test_start_time": test_start_time.isoformat(),
    "test_end_time": test_end_time.isoformat(),
    "total_test_time": str(test_end_time - test_start_time),
    "total_questions": len(test_cases),
    "successful_tests": len(successful_tests),
    "failed_tests": len(failed_tests),
    "success_rate": len(successful_tests) / len(test_cases) * 100,
    "avg_generation_time": avg_generation_time,
    "category_statistics": category_stats,
    "difficulty_statistics": difficulty_stats,
    "causal_editor_stats": final_stats,
    "debug_summary": debug_summary,
    "results": causal_results,
}

with open(causal_results_json_path, "w", encoding="utf-8") as f:
    json.dump(causal_test_summary, f, indent=2, ensure_ascii=False)

# 保存简化的CSV格式便于分析
causal_results_csv_path = result_path / "enhanced_dynamic_causal_editor_results.csv"
# 展平嵌套的统计信息以便CSV保存
flattened_results = []
for result in causal_results:
    flat_result = result.copy()
    # 展平causal_editor_stats
    if 'causal_editor_stats' in result:
        stats = result['causal_editor_stats']
        flat_result['conflict_detection_count'] = stats.get('conflict_detector_stats', {}).get('detection_count', 0)
        flat_result['conflict_count'] = stats.get('conflict_detector_stats', {}).get('conflict_count', 0)
        flat_result['edit_count'] = stats.get('counterfactual_editor_stats', {}).get('edit_count', 0)
        flat_result['successful_edits'] = stats.get('counterfactual_editor_stats', {}).get('successful_edits', 0)
        del flat_result['causal_editor_stats']  # 移除原始嵌套数据
    flattened_results.append(flat_result)

df_causal_results = pd.DataFrame(flattened_results)
df_causal_results.to_csv(causal_results_csv_path, index=False, encoding="utf-8")

# 生成增强的可读报告
causal_report_path = result_path / "enhanced_dynamic_causal_editor_report.txt"
with open(causal_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("增强版动态CausalEditor测试报告\n")
    f.write("=" * 80 + "\n")
    f.write(f"🤖 模型: {MODEL_NAME}\n")
    f.write(f"💻 设备: {DEVICE}\n")
    f.write(f"🔧 调试模式: {'启用' if DEBUG_MODE else '禁用'}\n")
    f.write(f"⏰ 测试时间: {test_start_time} - {test_end_time}\n")
    f.write(f"⏱️ 测试耗时: {test_end_time - test_start_time}\n")
    f.write(f"📊 问题总数: {len(test_cases)}\n")
    f.write(f"✅ 成功率: {len(successful_tests) / len(test_cases) * 100:.1f}%\n")
    f.write(f"⚡ 平均生成时间: {avg_generation_time:.3f}s\n\n")

    f.write("📈 按类别统计:\n")
    for category, stats in category_stats.items():
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        f.write(f"  📂 {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均时间: {stats['avg_time']:.3f}s\n")
    
    f.write("\n🎯 按难度统计:\n")
    for difficulty, stats in difficulty_stats.items():
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        f.write(f"  🎯 {difficulty}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均时间: {stats['avg_time']:.3f}s\n")

    f.write("\n🔧 动态CausalEditor统计信息:\n")
    f.write(
        f"  🔍 检测次数: {final_stats.get('conflict_detector_stats', {}).get('detection_count', 0)}\n"
    )
    f.write(
        f"  ⚠️ 冲突次数: {final_stats.get('conflict_detector_stats', {}).get('conflict_count', 0)}\n"
    )
    f.write(
        f"  ✏️ 编辑次数: {final_stats.get('counterfactual_editor_stats', {}).get('edit_count', 0)}\n"
    )
    f.write(
        f"  ✅ 成功编辑: {final_stats.get('counterfactual_editor_stats', {}).get('successful_edits', 0)}\n"
    )
    if final_stats.get('use_dynamic_mode'):
        f.write(f"  📚 动态索引大小: {final_stats.get('dynamic_index_size', 0)}\n")
    
    f.write("\n🐛 调试统计信息:\n")
    f.write(f"  📝 调试步骤总数: {debug_summary['total_steps']}\n")
    f.write(f"  ⏱️ 调试总时间: {debug_summary['total_time']:.3f}s\n")
    f.write(f"  ⚡ 冲突记录: {debug_summary['total_conflicts']}\n")
    f.write(f"  ✏️ 编辑记录: {debug_summary['total_edits']}\n")
    f.write(f"  📊 平均置信度: {debug_summary['avg_confidence']:.3f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("详细测试结果\n")
    f.write("=" * 80 + "\n")

    for i, result in enumerate(causal_results):
        status = "✅" if 'error' not in result else "❌"
        f.write(f"\n{status} 测试 {i + 1}: [{result.get('category', 'Unknown')}] [{result.get('difficulty', 'Unknown')}]\n")
        f.write(f"❓ 问题: {result['question']}\n")
        f.write(f"🎯 期望: {result['expected_answer']}\n")
        f.write(f"🤖 回答: {result['generated_answer']}\n")
        if 'error' not in result:
            f.write(f"⏱️ 生成时间: {result.get('generation_time', 0):.3f}s\n")
            f.write(f"📊 输入tokens: {result.get('input_tokens', 0)} | 输出tokens: {result.get('output_tokens', 0)}\n")
        f.write("-" * 60 + "\n")

# 保存调试详情
debug_details_path = result_path / "debug_details.json"
with open(debug_details_path, "w", encoding="utf-8") as f:
    json.dump(debug_summary, f, indent=2, ensure_ascii=False)

debug_monitor.end_step()

print(f"\n增强版动态CausalEditor测试结果已保存:")
print(f"  详细JSON: {causal_results_json_path}")
print(f"  CSV数据: {causal_results_csv_path}")
print(f"  可读报告: {causal_report_path}")
print(f"  调试详情: {debug_details_path}")

print("\n" + "="*60)
print("增强版动态因果编辑测试脚本已成功完成！")
print("动态模式实时进行候选筛选、指纹生成和冲突检测")
print("详细调试信息已记录，可用于单步调试观察数据流")
print("测试覆盖了多种场景和难度级别")
print("建议在IDE中设置断点进行深入调试分析")
print("="*60)
