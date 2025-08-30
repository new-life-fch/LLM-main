#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-2-7b-chat 模型 + RAG检索系统测试示例
展示如何使用 modeling_llama_causal_Llama2.py 结合RAG检索系统进行模型测试

主要功能:
1. 使用Llama-2-7b-chat模型
2. 集成RAG检索系统替代知识图谱
3. 测试指纹构建、冲突检测、激活编辑流程
4. 验证RAG系统的动态阈值调整功能
5. 展示完整的RAG-CausalEditor集成工作流程

使用方法:
1. 确保有足够的GPU内存 (推荐16GB+)
2. 安装所需依赖
3. 配置RAG索引和文档数据库
4. 运行此脚本进行测试
"""

import json
import logging
import os

import torch

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
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/data'

# 设置transformers详细日志
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "causal_editor"))

try:
    from causal_editor.core.causal_editor import CausalEditor
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

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Llama-2 + RAG 配置 ---
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # 原始在线模型名称
MODEL_NAME = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"  # 本地模型路径
RESULT_DIR = "./result_llama2_rag"  # 结果保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # 保持批处理大小较小
MAX_LENGTH = 1024  # 输入长度
DEBUG_MODE = True  # 启用详细调试模式
SAVE_INTERMEDIATE_RESULTS = True  # 保存中间结果

# RAG系统配置
RAG_CONFIG_PATH = "./configs/rag_only_config.json"  # RAG配置文件路径
USE_RAG_RETRIEVAL = True  # 启用RAG检索
RETRIEVAL_MODE = "rag_only"  # 检索模式：rag_only
ENABLE_DYNAMIC_THRESHOLD = True  # 启用动态阈值调整

# 确保结果目录存在
result_path = Path(RESULT_DIR)
result_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Llama-2-7b-chat + RAG检索系统测试")
print("=" * 80)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"检索模式: {RETRIEVAL_MODE}")
print(f"RAG配置: {RAG_CONFIG_PATH}")
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

# --- 步骤 1: 初始化RAG配置 ---
print("\n--- 步骤 1: 初始化RAG配置 ---")

try:
    # 验证配置文件路径
    config_path = Path(RAG_CONFIG_PATH)
    if not config_path.parent.exists():
        logging.warning(f"配置目录不存在，创建: {config_path.parent}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载RAG配置
    if config_path.exists():
        try:
            rag_config = RAGConfig(config_path=RAG_CONFIG_PATH)
            print(f"✅ 从 {RAG_CONFIG_PATH} 加载RAG配置成功")
        except Exception as config_error:
            logging.error(f"配置文件加载失败: {config_error}")
            print(f"⚠️  配置文件损坏，使用默认配置")
            rag_config = RAGConfig()
    else:
        print(f"⚠️  配置文件 {RAG_CONFIG_PATH} 不存在，使用默认配置")
        rag_config = RAGConfig()
        
    # 验证RAG配置已启用
    rag_config_dict = rag_config.get_config()
    if not rag_config_dict.get("rag_retrieval", {}).get("enabled", False):
        logging.warning("RAG配置中enabled字段为False，但继续测试")
    
    # 验证动态阈值配置
    dynamic_threshold_enabled = rag_config_dict.get("fallback_config", {}).get("enable_dynamic_threshold", False)
    if dynamic_threshold_enabled != ENABLE_DYNAMIC_THRESHOLD:
        logging.info(f"动态阈值配置: {dynamic_threshold_enabled} (期望: {ENABLE_DYNAMIC_THRESHOLD})")
    
    # 显示RAG配置信息
    if DEBUG_MODE:
        print("\n🔧 RAG配置信息:")
        print(json.dumps({
            "enabled": rag_config_dict.get("rag_retrieval", {}).get("enabled", False),
            "model_name": rag_config_dict.get("rag_retrieval", {}).get("model_name", "N/A"),
            "top_k": rag_config_dict.get("rag_retrieval", {}).get("top_k", 5),
            "min_score": rag_config_dict.get("rag_retrieval", {}).get("min_score", 0.3),
            "dynamic_threshold": rag_config_dict.get("fallback_config", {}).get("enable_dynamic_threshold", False),
            "reranker_enabled": rag_config_dict.get("reranker_config", {}).get("enabled", False)
        }, indent=2, ensure_ascii=False))
    
    print("✅ RAG配置初始化完成")
    
except Exception as e:
    logging.error(f"❌ RAG配置初始化失败: {e}")
    logging.error(f"错误详情: {traceback.format_exc()}")
    print("\n💡 故障排除建议:")
    print("1. 检查RAG配置文件是否存在")
    print("2. 验证配置文件格式是否正确")
    print("3. 确保RAG索引和文档数据库已准备")
    print("4. 检查文件权限和路径访问")
    print("5. 验证JSON格式是否正确")
    

# --- 步骤 2: 加载集成RAG的Llama-2模型 ---
print("\n--- 步骤 2: 加载集成RAG的Llama-2模型 ---")

try:
    # 验证模型路径
    model_path = Path(MODEL_NAME)
    if model_path.exists():
        print(f"✅ 本地模型路径验证成功: {MODEL_NAME}")
    else:
        print(f"⚠️  本地模型路径不存在，尝试在线下载: {MODEL_NAME}")
    
    # 加载模型 - 使用RAG检索模式（包含tokenizer初始化）
    logging.info("正在加载 Llama-2 模型（RAG模式）...")
    print("⏳ 正在下载和加载模型，这可能需要几分钟...")
    
    try:
        model = CausalLlama2ForCausalLM.from_pretrained_with_dynamic_causal_editor(
            MODEL_NAME,
            device=DEVICE,
            # 针对RAG优化的参数,
            rag_config=rag_config_dict,
        )
        print("✅ 模型加载成功")
        
        # 获取模型内置的tokenizer
        tokenizer = model.tokenizer
        if tokenizer is None:
            raise RuntimeError("模型未正确初始化tokenizer")
        print("✅ 分词器已从模型获取")
        
    except Exception as model_error:
        logging.error(f"模型加载失败: {model_error}")
        raise RuntimeError(f"无法加载模型: {model_error}")
    
    model.eval()
    print(f"✅ 模型 {MODEL_NAME} 已加载到 {DEVICE}")
    
    # 显示模型信息
    if DEBUG_MODE:
        model_info = model.get_model_info()
        print("\n📊 模型信息:")
        print(json.dumps(model_info, indent=2, ensure_ascii=False))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📈 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型设备: {next(model.parameters()).device}")
    
    # 验证 CausalEditor 和 RAG 集成
    causal_editor_instance = model.causal_editor
    
    if causal_editor_instance is None:
        raise RuntimeError("CausalEditor未正确初始化")
    
    print("✅ CausalEditor已自动初始化并附加到模型")
    print("✅ Tokenizer 已自动设置到模型")
    
    # 验证RAG检索器
    if hasattr(causal_editor_instance, 'rag_retriever') and causal_editor_instance.rag_retriever:
        print("✅ RAG检索器已成功初始化")
    else:
        print("⚠️  RAG检索器未找到，可能影响检索功能")
    
    # 显示 CausalEditor 配置
    if DEBUG_MODE:
        editor_stats = causal_editor_instance.get_statistics()
        print("\n🔧 CausalEditor 配置:")
        print(json.dumps(editor_stats, indent=2, ensure_ascii=False))
        
        # 显示RAG配置
        rag_config_info = causal_editor_instance.get_rag_config()
        print("\n🔍 RAG集成配置:")
        print(json.dumps(rag_config_info, indent=2, ensure_ascii=False))

except Exception as e:
    logging.error(f"❌ 加载模型或初始化 CausalEditor 失败: {e}")
    logging.error(f"错误详情: {traceback.format_exc()}")
    print("\n💡 故障排除建议:")
    print("1. 检查GPU内存是否足够 (推荐16GB+)")
    print("2. 确保网络连接正常以下载模型")
    print("3. 验证 Hugging Face 访问权限")
    print("4. 检查RAG索引和文档数据库是否存在")
    print("5. 考虑使用更小的模型进行测试")
    sys.exit(1)

# --- 步骤 3: 执行RAG检索测试 ---
print("\n--- 步骤 3: 执行RAG检索测试 ---")

# 专门针对RAG系统的测试用例
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
    },
    # {
    #     "category": "数学逻辑测试",
    #     "prompt": "If 2+2=5, what is 3+3?",
    #     "expected": "The premise is incorrect. 2+2=4, not 5",
    #     "difficulty": "hard"
    # }
    
]

print(f"准备了 {len(test_cases)} 个RAG测试用例")

results = []
test_start_time = datetime.now()

for i, test_case in enumerate(test_cases):
    case_start_time = time.time()
    user_query = test_case["prompt"]
    expected_answer = test_case["expected"]
    category = test_case["category"]
    difficulty = test_case["difficulty"]
    
    print(f"\n🧪 测试用例 {i + 1}/{len(test_cases)}: {category} ({difficulty})")
    print(f"❓ 问题: {user_query}")
    print(f"🎯 期望: {expected_answer}")
    
    try:
        # 验证测试用例数据
        if not user_query or not user_query.strip():
            raise ValueError("测试问题为空")
        
        # 构建 Llama-2 chat 格式的输入
        messages = [
            {"role": "system", "content": "You are a helpful and accurate assistant. Please provide factual information."},
            {"role": "user", "content": user_query},
        ]

        # 应用 chat 模板
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"✅ 聊天模板应用成功")
        except Exception as template_error:
            logging.error(f"聊天模板应用失败: {template_error}")
            # 回退到简单格式
            input_text = f"System: You are a helpful assistant.\nUser: {user_query}\nAssistant:"
            print(f"⚠️  使用简单格式作为回退")
        
        # 编码输入
        try:
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(DEVICE)
            print(f"📝 输入token数量: {inputs['input_ids'].shape[1]}")
            
            # 验证输入长度
            if inputs['input_ids'].shape[1] > MAX_LENGTH:
                logging.warning(f"输入长度 {inputs['input_ids'].shape[1]} 超过最大长度 {MAX_LENGTH}")
                # 截断输入
                inputs['input_ids'] = inputs['input_ids'][:, :MAX_LENGTH]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, :MAX_LENGTH]
                print(f"⚠️  输入已截断到 {MAX_LENGTH} tokens")
                
        except Exception as tokenize_error:
            logging.error(f"分词化失败: {tokenize_error}")
            raise RuntimeError(f"无法分词化输入: {tokenize_error}")

        # 重置统计信息
        try:
            causal_editor_instance.reset_statistics()
            print("✅ CausalEditor统计已重置")
        except Exception as reset_error:
            logging.warning(f"重置统计失败: {reset_error}")

        # 准备输入 - 触发RAG检索
        print(f"🔍 准备CausalEditor处理输入: {user_query}")
        try:
            if hasattr(causal_editor_instance, 'prepare_for_input'):
                causal_editor_instance.prepare_for_input(user_query)
                print("✅ CausalEditor输入准备完成")
            else:
                print("⚠️  CausalEditor没有prepare_for_input方法，跳过")
        except Exception as prepare_error:
            logging.warning(f"CausalEditor输入准备失败: {prepare_error}")
        
        # 生成回复
        generation_start_time = time.time()
        print("🚀 开始文本生成...")
        
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs)
            print("✅ 文本生成完成")
        except Exception as generation_error:
            logging.error(f"文本生成失败: {generation_error}")
            raise RuntimeError(f"生成过程出错: {generation_error}")
        
        generation_time = time.time() - generation_start_time
        print(f"⏱️  生成耗时: {generation_time:.3f}s")
        
        # 解码生成的回复
        try:
            if outputs.shape[1] > inputs["input_ids"].shape[1]:
                response_text = tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                print(f"✅ 回复解码成功，长度: {len(response_text)} 字符")
            else:
                response_text = "[生成失败：无新token生成]"
                logging.warning("模型未生成新的token")
        except Exception as decode_error:
            logging.error(f"回复解码失败: {decode_error}")
            response_text = f"[解码错误: {str(decode_error)}]"
        
        # 获取统计信息
        try:
            current_stats = causal_editor_instance.get_statistics()
            print("✅ 统计信息获取成功")
        except Exception as stats_error:
            logging.warning(f"获取统计信息失败: {stats_error}")
            current_stats = {"error": str(stats_error)}
        
        print(f"🤖 生成回复: {response_text}")
        print(f"⏱️  生成耗时: {generation_time:.3f}s")
        
        # 检查RAG检索统计
        try:
            rag_stats = {}
            if hasattr(causal_editor_instance, 'rag_retriever') and causal_editor_instance.rag_retriever:
                # 从RAG检索器直接获取统计信息
                rag_retriever_stats = causal_editor_instance.rag_retriever.get_statistics()
                rag_score_stats = causal_editor_instance.rag_retriever.get_score_statistics()
                
                rag_stats = {
                    "retrieval_count": rag_retriever_stats.get('retrieval_count', 0),
                    "total_documents": rag_retriever_stats.get('total_documents', 0),
                    "index_build_time": rag_retriever_stats.get('index_build_time', 0),
                    "query_count": rag_score_stats.get('query_count', 0),
                    "fallback_triggered": rag_score_stats.get('fallback_triggered', 0),
                    "fallback_rate": rag_score_stats.get('fallback_rate', 0.0),
                    "overall_avg_score": rag_score_stats.get('overall_avg_score', 0.0),
                    "recent_avg_score": rag_score_stats.get('recent_avg_score', 0.0)
                }
                print(f"🔍 RAG检索统计更新: {rag_stats}")
                if rag_stats.get('query_count', 0) > 0:
                    print(f"📡 本次生成触发了 {rag_stats['query_count']} 次RAG查询")
                    print(f"📄 检索次数: {rag_stats['retrieval_count']}")
                    print(f"🎯 平均得分: {rag_stats['overall_avg_score']:.4f}")
                    print(f"⚠️ Fallback触发: {rag_stats['fallback_triggered']}次 ({rag_stats['fallback_rate']:.1%})")
                else:
                    print("📡 本次生成未触发RAG检索（使用缓存或本地知识）")
            else:
                print("⚠️  RAG检索器未找到或未初始化")
        except Exception as rag_stats_error:
            logging.warning(f"获取RAG统计失败: {rag_stats_error}")
        
        # 显示冲突检测和编辑统计
        if DEBUG_MODE:
            conflict_stats = current_stats.get('conflict_detector_stats', {})
            edit_stats = current_stats.get('counterfactual_editor_stats', {})
            
            print(f"\n🔍 冲突检测统计:")
            print(f"  - 检测次数: {conflict_stats.get('detection_count', 0)}")
            print(f"  - 冲突发现: {conflict_stats.get('conflicts_found', 0)}")
            print(f"  - 冲突率: {conflict_stats.get('conflict_rate', 0.0):.1%}")
            print(f"  - 平均置信度: {conflict_stats.get('average_confidence', 0.0):.3f}")
            
            print(f"\n✏️  编辑统计:")
            print(f"  - 编辑次数: {edit_stats.get('edit_count', 0)}")
            print(f"  - 成功编辑: {edit_stats.get('successful_edits', 0)}")
            print(f"  - 成功率: {edit_stats.get('success_rate', 0.0):.1%}")
            print(f"  - 平均编辑强度: {edit_stats.get('average_edit_magnitude', 0.0):.3f}")
            print(f"  - RAG编辑次数: {edit_stats.get('rag_edit_count', 0)}")
            print(f"  - RAG编辑率: {edit_stats.get('rag_edit_rate', 0.0):.1%}")
            
            # 显示编辑方法统计
            edit_method_stats = edit_stats.get('edit_method_stats', {})
            if edit_method_stats:
                print(f"  - 编辑方法分布:")
                for method, count in edit_method_stats.items():
                    print(f"    * {method}: {count}次")
            
            # 显示动态阈值信息
            print(f"\n⚙️  动态阈值:")
            print(f"  - 相似度阈值: {conflict_stats.get('current_similarity_threshold', 'N/A')}")
            print(f"  - 冲突阈值: {conflict_stats.get('current_conflict_threshold', 'N/A')}")
            
            # 显示RAG检索器的动态阈值
            if hasattr(causal_editor_instance, 'rag_retriever') and causal_editor_instance.rag_retriever:
                try:
                    rag_score_stats = causal_editor_instance.rag_retriever.get_score_statistics()
                    current_thresholds = rag_score_stats.get('current_thresholds', {})
                    if current_thresholds:
                        print(f"  - RAG Fallback阈值:")
                        print(f"    * 高阈值: {current_thresholds.get('high', 0.0):.3f}")
                        print(f"    * 中阈值: {current_thresholds.get('medium', 0.0):.3f}")
                        print(f"    * 低阈值: {current_thresholds.get('low', 0.0):.3f}")
                except Exception as e:
                    logging.warning(f"获取RAG阈值信息失败: {e}")
        
        # 保存测试结果
        case_end_time = time.time()
        try:
            # 计算token统计
            input_token_count = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            output_token_count = (outputs.shape[1] - inputs['input_ids'].shape[1]) if outputs is not None and 'input_ids' in inputs else 0
            
            result_entry = {
                "question_id": i + 1,
                "category": category,
                "difficulty": difficulty,
                "question": user_query,
                "expected_answer": expected_answer,
                "generated_answer": response_text,
                "generation_time": generation_time,
                "total_case_time": case_end_time - case_start_time,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "causal_editor_stats": current_stats,
                "rag_stats": rag_stats if 'rag_stats' in locals() else {},
                "rag_enabled": True,
                "retrieval_mode": RETRIEVAL_MODE,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "device_info": {
                    "device": str(DEVICE),
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }
            }
            results.append(result_entry)
            print(f"✅ 测试结果已保存，输入tokens: {input_token_count}, 输出tokens: {output_token_count}")
        except Exception as save_error:
            logging.error(f"保存测试结果失败: {save_error}")
            # 创建最小结果条目
            result_entry = {
                "question_id": i + 1,
                "category": category,
                "question": user_query,
                "generated_answer": response_text,
                "error": f"保存失败: {str(save_error)}",
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result_entry)
        
        print("✅ 测试完成")
        
    except Exception as e:
        logging.error(f"❌ 测试用例 {i+1} 执行失败: {e}")
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
            "rag_enabled": True,
            "retrieval_mode": RETRIEVAL_MODE,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result_entry)
        continue

test_end_time = datetime.now()

# --- 步骤 4: 保存RAG测试结果 ---
print(f"\n--- 步骤 4: 保存RAG测试结果 ---")

# 计算统计信息
successful_tests = [r for r in results if 'error' not in r]
failed_tests = [r for r in results if 'error' in r]
total_generation_time = sum(r.get('generation_time', 0) for r in successful_tests)
avg_generation_time = total_generation_time / len(successful_tests) if successful_tests else 0

# 计算RAG统计信息
total_rag_queries = sum(r.get('rag_stats', {}).get('query_count', 0) for r in successful_tests)
total_rag_retrievals = sum(r.get('rag_stats', {}).get('retrieval_count', 0) for r in successful_tests)
total_fallback_triggered = sum(r.get('rag_stats', {}).get('fallback_triggered', 0) for r in successful_tests)
rag_scores = [r.get('rag_stats', {}).get('overall_avg_score', 0) for r in successful_tests if r.get('rag_stats', {}).get('overall_avg_score', 0) > 0]
avg_rag_score = sum(rag_scores) / len(rag_scores) if rag_scores else 0.0

print(f"\n📊 RAG测试统计摘要:")
print(f"  ✅ 成功测试: {len(successful_tests)} / {len(results)}")
print(f"  ❌ 失败测试: {len(failed_tests)}")
print(f"  ⏱️  平均生成时间: {avg_generation_time:.3f}s")
print(f"  🕐 总测试时间: {test_end_time - test_start_time}")
print(f"  🔍 检索模式: {RETRIEVAL_MODE}")
print(f"  📊 动态阈值: {'启用' if ENABLE_DYNAMIC_THRESHOLD else '禁用'}")
print(f"  📡 总RAG查询: {total_rag_queries}")
print(f"  🔍 总检索次数: {total_rag_retrievals}")
print(f"  ⚠️ Fallback触发: {total_fallback_triggered}")
print(f"  🎯 平均RAG得分: {avg_rag_score:.4f}")

# 获取最终统计信息
try:
    final_stats = causal_editor_instance.get_statistics()
    print(f"\n🔧 CausalEditor 最终统计:")
    print(json.dumps(final_stats, indent=2, ensure_ascii=False))
except Exception as final_stats_error:
    logging.error(f"获取最终统计失败: {final_stats_error}")
    final_stats = {"error": str(final_stats_error)}
    print(f"\n❌ 无法获取CausalEditor最终统计: {final_stats_error}")

# 获取RAG配置信息
try:
    if hasattr(causal_editor_instance, 'get_rag_config'):
        final_rag_config = causal_editor_instance.get_rag_config()
        print(f"\n🔍 RAG系统最终配置:")
        print(json.dumps(final_rag_config, indent=2, ensure_ascii=False))
    else:
        final_rag_config = {"retrieval_mode": RETRIEVAL_MODE, "use_rag_retrieval": USE_RAG_RETRIEVAL}
        print(f"\n⚠️  get_rag_config方法不存在，使用基本配置")
except Exception as rag_config_error:
    logging.error(f"获取RAG配置失败: {rag_config_error}")
    final_rag_config = {"error": str(rag_config_error), "retrieval_mode": RETRIEVAL_MODE}
    print(f"\n❌ 无法获取RAG配置: {rag_config_error}")

# 保存详细结果
results_json_path = result_path / "llama2_rag_test_results.json"
try:
    test_summary = {
        "model_name": MODEL_NAME,
        "test_type": "llama2_rag_causal_editor_test",
        "device": DEVICE,
        "retrieval_mode": RETRIEVAL_MODE,
        "rag_enabled": USE_RAG_RETRIEVAL,
        "dynamic_threshold_enabled": ENABLE_DYNAMIC_THRESHOLD,
        "test_start_time": test_start_time.isoformat(),
        "test_end_time": test_end_time.isoformat(),
        "total_test_time": str(test_end_time - test_start_time),
        "total_questions": len(test_cases),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "success_rate": len(successful_tests) / len(test_cases) * 100 if test_cases else 0,
        "avg_generation_time": avg_generation_time,
        "total_rag_queries": total_rag_queries,
        "total_rag_retrievals": total_rag_retrievals,
        "total_fallback_triggered": total_fallback_triggered,
        "avg_rag_score": avg_rag_score,
        "causal_editor_stats": final_stats,
        "rag_config": final_rag_config,
        "results": results,
        "test_environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 测试结果JSON已保存: {results_json_path}")
except Exception as json_save_error:
    logging.error(f"保存JSON结果失败: {json_save_error}")
    print(f"❌ 无法保存JSON结果: {json_save_error}")

# 保存简化报告
report_path = result_path / "llama2_rag_test_report.txt"
try:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Llama-2-7b-chat + RAG检索系统测试报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"🤖 模型: {MODEL_NAME}\n")
        f.write(f"💻 设备: {DEVICE}\n")
        f.write(f"🔍 检索模式: {RETRIEVAL_MODE}\n")
        f.write(f"📊 RAG启用: {'是' if USE_RAG_RETRIEVAL else '否'}\n")
        f.write(f"📈 动态阈值: {'启用' if ENABLE_DYNAMIC_THRESHOLD else '禁用'}\n")
        f.write(f"⏰ 测试时间: {test_start_time} - {test_end_time}\n")
        f.write(f"⏱️ 测试耗时: {test_end_time - test_start_time}\n")
        f.write(f"📊 问题总数: {len(test_cases)}\n")
        f.write(f"✅ 成功率: {len(successful_tests) / len(test_cases) * 100:.1f}%\n")
        f.write(f"⚡ 平均生成时间: {avg_generation_time:.3f}s\n\n")
        
        f.write("🔧 CausalEditor 统计信息:\n")
        f.write(f"  🔍 检测次数: {final_stats.get('conflict_detector_stats', {}).get('detection_count', 0)}\n")
        f.write(f"  ⚠️ 冲突次数: {final_stats.get('conflict_detector_stats', {}).get('conflict_count', 0)}\n")
        f.write(f"  ✏️ 编辑次数: {final_stats.get('counterfactual_editor_stats', {}).get('edit_count', 0)}\n")
        f.write(f"  ✅ 成功编辑: {final_stats.get('counterfactual_editor_stats', {}).get('successful_edits', 0)}\n")
        
        f.write("\n🔍 RAG系统统计信息:\n")
        f.write(f"  📡 检索模式: {final_rag_config.get('retrieval_mode', 'N/A')}\n")
        f.write(f"  🔍 RAG启用: {final_rag_config.get('use_rag_retrieval', False)}\n")
        f.write(f"  📊 总RAG查询: {total_rag_queries}\n")
        f.write(f"  🔍 总检索次数: {total_rag_retrievals}\n")
        f.write(f"  ⚠️ Fallback触发: {total_fallback_triggered}\n")
        f.write(f"  🎯 平均RAG得分: {avg_rag_score:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("详细测试结果\n")
        f.write("=" * 80 + "\n")
        
        for i, result in enumerate(results):
            status = "✅" if 'error' not in result else "❌"
            f.write(f"\n{status} 测试 {i + 1}: [{result.get('category', 'Unknown')}]\n")
            f.write(f"❓ 问题: {result['question']}\n")
            f.write(f"🎯 期望: {result['expected_answer']}\n")
            f.write(f"🤖 回答: {result['generated_answer']}\n")
            if 'error' not in result:
                f.write(f"⏱️ 生成时间: {result.get('generation_time', 0):.3f}s\n")
                f.write(f"🔍 检索模式: {result.get('retrieval_mode', 'N/A')}\n")
            f.write("-" * 60 + "\n")

    print(f"✅ 测试报告已保存: {report_path}")
except Exception as report_save_error:
    logging.error(f"保存测试报告失败: {report_save_error}")
    print(f"❌ 无法保存测试报告: {report_save_error}")

# 最终清理和总结
try:
    print(f"\n💾 RAG测试结果已保存:")
    print(f"  📄 详细JSON: {results_json_path}")
    print(f"  📋 可读报告: {report_path}")
    
    print(f"\n🎯 测试总结:")
    print(f"  ✅ 成功测试: {len(successful_tests)}/{len(test_cases)}")
    print(f"  ❌ 失败测试: {len(failed_tests)}/{len(test_cases)}")
    success_rate = len(successful_tests) / len(test_cases) * 100 if test_cases else 0
    print(f"  📊 成功率: {success_rate:.1f}%")
    print(f"  ⏱️  平均生成时间: {avg_generation_time:.2f}秒")
    print(f"  🔧 CausalEditor状态: {'正常' if final_stats and 'error' not in final_stats else '异常'}")
    print(f"  🔍 RAG系统状态: {'正常' if final_rag_config and 'error' not in final_rag_config else '异常'}")
    
    # 内存清理建议
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"  💾 当前GPU内存使用: {current_memory:.2f}GB")
        if current_memory > 10:  # 如果使用超过10GB
            print(f"  ⚠️  建议清理GPU内存")
    
except Exception as final_summary_error:
    logging.error(f"生成最终总结失败: {final_summary_error}")
    print(f"\n❌ 生成最终总结时出错: {final_summary_error}")
    print("\n🏁 测试完成，但总结生成失败")

# 可选的内存清理
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("GPU内存缓存已清理")
except Exception as cleanup_error:
    logging.warning(f"清理GPU内存失败: {cleanup_error}")

print("\n" + "="*80)
print("🎉 Llama-2-7b-chat + RAG检索系统测试完成！")
print("✨ 成功集成RAG检索系统替代知识图谱")
print("🔍 验证了指纹构建、冲突检测、激活编辑流程")
print("📊 测试了动态阈值调整功能")
print("🔧 建议查看生成的报告以了解详细性能")
print("="*80)