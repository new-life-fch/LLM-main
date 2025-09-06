#!/usr/bin/env python3
"""
使用CausalEditor Pipeline的示例脚本

这个脚本展示了如何使用自定义的CausalEditorPipeline来处理TruthfulQA数据集。
Pipeline集成了RAG检索和因果编辑功能，基于FlashRAG框架构建。
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from causal_editor_pipeline import CausalEditorPipeline
from causal_editor.utils.utils import sample_dataset


def main():
    """主函数"""
    print("=" * 80)
    print("CausalEditor Pipeline 运行示例")
    print("=" * 80)
    
    # --- 配置参数 ---
    MODEL_NAME = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/Llama-3.1-8B-Instruct"
    RETRIEVAL_CONFIG_PATH = "./configs/retrieval_config.yaml"
    CAUSAL_EDITOR_CONFIG_PATH = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/configs/causal_editor.json"
    DATASET_NAME = "hotpot"  # 使用Natural Questions数据集
    DATA_PATH = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/dataset/hotpot_dev.jsonl"
    RESULT_DIR = "./result/result_causal_editor_pipeline/hotpot"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 4096
    MAX_NEW_TOKENS = 256
    
    print(f"模型路径: {MODEL_NAME}")
    print(f"数据集: {DATASET_NAME}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"设备: {DEVICE}")
    print("=" * 80)
    
    # --- 检查文件是否存在 ---
    print("\n--- 检查必要文件 ---")
    
    if not Path(MODEL_NAME).exists():
        print(f"❌ 模型路径不存在: {MODEL_NAME}")
        print("请确保模型已下载到指定路径")
        return
    print(f"✅ 模型路径存在: {MODEL_NAME}")
    
    if not Path(RETRIEVAL_CONFIG_PATH).exists():
        print(f"❌ 检索配置文件不存在: {RETRIEVAL_CONFIG_PATH}")
        return
    print(f"✅ 检索配置文件存在: {RETRIEVAL_CONFIG_PATH}")
    
    if not Path(CAUSAL_EDITOR_CONFIG_PATH).exists():
        print(f"❌ CausalEditor配置文件不存在: {CAUSAL_EDITOR_CONFIG_PATH}")
        return
    print(f"✅ CausalEditor配置文件存在: {CAUSAL_EDITOR_CONFIG_PATH}")
    
    print(f"✅ 将使用FlashRAG数据集: {DATASET_NAME}")
    
    # --- 创建FlashRAG配置 ---
    print("\n--- 创建FlashRAG配置 ---")
    # 加载FlashRAG配置
    from flashrag.config import Config
    config = Config(config_file_path=RETRIEVAL_CONFIG_PATH)
    
    print("✅ FlashRAG配置创建完成")
    
    try:
        # --- 初始化Pipeline ---
        print("\n--- 初始化CausalEditor Pipeline ---")
        
        pipeline = CausalEditorPipeline(
            config=config,
            model_name=MODEL_NAME,
            causal_editor_config_path=CAUSAL_EDITOR_CONFIG_PATH,
            max_length=MAX_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE
        )
        
        print("✅ CausalEditor Pipeline初始化成功")
        
        # --- 加载数据集 ---
        print(f"\n--- 加载{DATASET_NAME}数据集 ---")
        test_data = sample_dataset(DATASET_NAME, DATA_PATH, 20)
        
        from flashrag.dataset import Dataset
        dataset = Dataset(config, data = test_data)
        print(f"✅ 成功加载 {len(dataset)} 个问题")
        
        # 显示前几个问题作为示例
        print("\n前3个问题示例:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"  {i+1}. {item.question[:80]}...")
        
        # --- 运行Pipeline ---
        print("\n--- 运行CausalEditor Pipeline ---")
        print("开始处理问题...")
        
        result_dataset = pipeline.run(dataset, do_eval=True)
        
        print("✅ Pipeline运行完成")
        
        # --- 保存结果 ---
        print("\n--- 保存结果 ---")
        
        saved_files = pipeline.save_results(result_dataset, RESULT_DIR)
        
        print("✅ 结果保存完成")
        
        # --- 显示结果摘要 ---
        print("\n" + "=" * 80)
        print("CausalEditor Pipeline 运行完成！")
        print("=" * 80)
        
        # 统计信息
        total_questions = len(result_dataset)
        successful_questions = sum(1 for item in result_dataset if hasattr(item, 'pred') and item.pred)
        success_rate = successful_questions / total_questions * 100 if total_questions > 0 else 0
        
        print(f"📊 处理统计:")
        print(f"   总问题数: {total_questions}")
        print(f"   成功处理: {successful_questions}")
        print(f"   成功率: {success_rate:.1f}%")
        
        # 显示评估结果
        try:
            if hasattr(result_dataset, 'eval_results') and result_dataset.eval_results:
                print(f"\n📈 评估结果:")
                for metric, score in result_dataset.eval_results.items():
                    print(f"   {metric}: {score:.4f}")
            else:
                print(f"\n📈 评估结果: 未进行评估 (do_eval=False)")
        except (KeyError, AttributeError) as e:
            print(f"\n📈 评估结果: 未进行评估 (do_eval=False)")
            print(f"   注意: 如需评估结果，请设置 do_eval=True")
        
        print(f"\n📁 生成的文件:")
        for file_type, file_path in saved_files.items():
            print(f"   - {file_type.replace('_', ' ').title()}: {file_path}")
        
        # 显示几个生成的答案示例
        print(f"\n📝 生成答案示例:")
        for i in range(min(3, len(result_dataset))):
            item = result_dataset[i]
            answer = getattr(item, 'pred', 'N/A')
            has_answer = hasattr(item, 'pred') and item.pred
            status = "✅" if has_answer else "❌"
            print(f"   {i+1}. {status} Q: {item.question[:60]}...")
            print(f"      A: {answer[:100]}...")
            print()
        
        print("=" * 80)
        print("🎉 所有任务完成！")
        
    except Exception as e:
        print(f"\n❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 检查RAG配置文件是否正确")
        print("3. 检查CUDA内存是否足够")
        print("4. 检查所有依赖是否正确安装")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)