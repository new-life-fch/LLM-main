import json
import logging
import os
import csv
import torch
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 设置详细的日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 设置 Hugging Face 缓存目录
os.environ['HF_HOME'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data'
os.environ['TRANSFORMERS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp" 
os.environ['HF_DATASETS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp"

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "causal_editor"))
sys.path.insert(0, str(project_root / "FlashRag" / "FlashRAG"))

# 导入FlashRAG组件
from flashrag.pipeline import BasicPipeline
from flashrag.config import Config
from flashrag.utils import get_retriever
from flashrag.prompt import PromptTemplate

# 导入项目组件
from causal_editor.core.causal_editor import CausalEditor
from modeling_llama_causal.modeling_llama_causal_Llama2 import CausalLlama2ForCausalLM
from modeling_llama_causal.modeling_llama_causal_Llama3_1 import CausalLlama31ForCausalLM
from causal_editor.dynamic.rag_config import RAGConfig
from causal_editor.dynamic.fingerprint_builder import DynamicFingerprintBuilder
from causal_editor.core.conflict_detector import CausalConflictDetector
from causal_editor.core.counterfactual_editor import CounterfactualEditor


class CausalEditorPipeline(BasicPipeline):
    """
    自定义的CausalEditor Pipeline，集成RAG检索和因果编辑功能
    """
    
    def __init__(self, config, prompt_template=None, model_name=None, causal_editor_config_path=None, 
                 max_length=4096, max_new_tokens=256, device=None,generator=None, retriever=None):
        """
        初始化CausalEditor Pipeline
        
        Args:
            config: FlashRAG配置对象 (从retrieval_config.yaml创建)
            prompt_template: 提示模板
            model_name: 模型路径
            causal_editor_config_path: CausalEditor框架配置文件路径 (JSON)
            max_length: 最大输入长度
            max_new_tokens: 最大生成token数
            device: 设备
        """
        super().__init__(config, prompt_template)
        
        # 设置参数
        self.model_name = model_name or "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/Llama-3.1-8B-Instruct"
        self.causal_editor_config_path = causal_editor_config_path or "./configs/causal_editor.json"
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.flashrag_config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self._initialize_causal_editor_config()
        self._initialize_model()
        self._initialize_retriever()
        
        print(f"✅ CausalEditor Pipeline 初始化完成")
        print(f"   模型: {self.model_name}")
        print(f"   设备: {self.device}")
        print(f"   CausalEditor配置: {self.causal_editor_config_path}")
    
    def _initialize_causal_editor_config(self):
        """
        初始化CausalEditor配置文件
        """
        print("\n--- 初始化CausalEditor配置 ---")
        
        # 加载CausalEditor配置 (JSON)
        with open(self.causal_editor_config_path, 'r', encoding='utf-8') as f:
            self.causal_editor_config = json.load(f)
        
        print("✅ CausalEditor配置初始化完成")
    
    def _initialize_model(self):
        """初始化集成CausalEditor的模型"""
        print("\n--- 加载集成RAG的Llama-2模型 ---")
        
        try:
            print("⏳ 正在加载模型...")
            
            self.model = CausalLlama31ForCausalLM.from_pretrained_with_dynamic_causal_editor(
                self.model_name,
                device=self.device,
                rag_config=self.causal_editor_config
            )
            print("✅ 模型加载成功")
            
            # 获取模型内置的tokenizer
            self.tokenizer = self.model.tokenizer
            if self.tokenizer is None:
                raise RuntimeError("模型未正确初始化tokenizer")
            print("✅ 分词器已从模型获取")
            
        except Exception as model_error:
            print(f"❌ 模型加载失败: {model_error}")
            raise model_error
        
        self.model.eval()
        print(f"✅ 模型 {self.model_name} 已加载到 {self.device}")
        
        # 验证 CausalEditor 和 RAG 集成
        self.causal_editor_instance = self.model.causal_editor
        
        if self.causal_editor_instance is None:
            raise RuntimeError("CausalEditor未正确初始化")
        
        print("✅ CausalEditor已自动初始化并附加到模型")
    
    def _initialize_retriever(self):
        """初始化检索器"""
        print("\n--- 初始化检索器 ---")
        
        # 使用FlashRAG的检索器
        self.retriever = get_retriever(self.flashrag_config)
        print("✅ 检索器初始化完成")
    
    def _prepare_input(self, question: str) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        # 构建 Llama-2 chat 格式的输入
        messages = [
            {"role": "system", "content": "Answer the question and only give me one answer without outputting any other words."},
            {"role": "user", "content": question},
        ]

        # 应用 chat 模板
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 回退到简单格式
            input_text = f"System: You are a helpful assistant.\nUser: {question}\nAssistant:"
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,      # 单样本不需要padding
            truncation=True,    # 防止序列过长
            max_length=self.max_length
        ).to(self.device)
        
        # 验证输入长度并截断
        if inputs['input_ids'].shape[1] > self.max_length:
            inputs['input_ids'] = inputs['input_ids'][:, :self.max_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_length]
        
        return inputs
    
    def _generate_answer(self, question: str) -> Dict[str, Any]:
        """生成单个问题的答案"""
        try:
            # 生成回复
            generation_start_time = time.time()
            # 准备输入
            inputs = self._prepare_input(question)
            

            # 准备输入 - 触发RAG检索
            try:
                if hasattr(self.causal_editor_instance, 'prepare_for_input'):
                    self.causal_editor_instance.prepare_for_input(question, self.retriever)
            except Exception:
                pass
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                )
            
            generation_time = time.time() - generation_start_time
            self.causal_editor_instance.finish_generation()
            
            # 解码生成的回复
            if outputs.shape[1] > inputs["input_ids"].shape[1]:
                response_text = self.tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            else:
                response_text = "[生成失败：无新token生成]"
            
            return {
                "answer": response_text,
                "generation_time": generation_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "answer": f"ERROR: {str(e)}",
                "generation_time": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        运行pipeline的主要方法
        
        Args:
            dataset: 数据集对象，包含question属性
            do_eval: 是否进行评估
            pred_process_fun: 预测后处理函数
        
        Returns:
            处理后的数据集
        """
        print("\n--- 开始CausalEditor Pipeline推理 ---")
        
        pred_answer_list = []
        generation_times = []
        success_count = 0

        # 重置统计信息
        try:
            self.causal_editor_instance.reset_statistics()
        except Exception:
            pass
        
        # 处理每个问题
        for item in tqdm(dataset, desc="CausalEditor推理中"):
            question = item.question
            
            # 生成答案
            result = self._generate_answer(question)
            
            pred_answer_list.append(result["answer"])
            generation_times.append(result["generation_time"])
            
            if result["success"]:
                success_count += 1
            
            # 更新item的输出
            item.update_output("pred", result["answer"])
            item.update_output("generation_time", result["generation_time"])
            item.update_output("success", result["success"])
            if result["error"]:
                item.update_output("error", result["error"])
        
        # 更新数据集输出
        dataset.update_output("pred", pred_answer_list)
        dataset.update_output("generation_times", generation_times)
        
        # 打印统计信息
        total_questions = len(dataset)
        avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
        success_rate = success_count / total_questions * 100 if total_questions > 0 else 0
        
        print(f"\n--- CausalEditor Pipeline 统计 ---")
        print(f"总问题数: {total_questions}")
        print(f"成功生成: {success_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均生成时间: {avg_generation_time:.3f}s")
        
        # 评估 - 使用BasicPipeline的evaluate方法
        print("\n开始评估结果...")
        try:
            if do_eval:
                # 调用evaluator获取评估结果
                eval_results = self.evaluator.evaluate(dataset)
                
                print("评估完成，结果如下:")
                for metric, score in eval_results.items():
                    print(f"  {metric}: {score:.4f}")
                    
                # 将评估结果保存到数据集中
                dataset.eval_results = eval_results
            
            # 保存检索缓存（如果需要）
            if self.save_retrieval_cache and hasattr(self, 'retriever'):
                self.retriever._save_cache()
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            if do_eval:
                dataset.eval_results = {}
        
        return dataset
    
    def _collect_causal_editor_statistics(self) -> Dict[str, Any]:
        """
        收集CausalEditor运行时的统计信息
        
        Returns:
            包含冲突检测和编辑统计信息的字典
        """
        statistics = {
            "causal_editor_enabled": self.causal_editor_instance is not None,
            "rag_config": self.causal_editor_config if hasattr(self, 'causal_editor_config') else {},
            "conflict_detection": {
                "total_detections": 0,
                "conflicts_found": 0,
                "conflict_rate": 0.0,
                "avg_confidence": 0.0
            },
            "editing_operations": {
                "total_edits": 0,
                "successful_edits": 0,
                "edit_success_rate": 0.0,
                "layers_edited": []
            },
            "rag_retrieval": {
                "retrieval_enabled": False,
                "total_retrievals": 0,
                "avg_retrieved_docs": 0.0,
                "fingerprint_cache_hits": 0
            },
            "performance": {
                "avg_detection_time": 0.0,
                "avg_edit_time": 0.0,
                "memory_usage": 0.0
            }
        }
        
        try:
            if self.causal_editor_instance:
                # 获取真实的运行时统计数据
                real_stats = self.causal_editor_instance.get_statistics()
                
                # 收集基本配置信息
                if hasattr(self.causal_editor_instance, 'rag_config'):
                    statistics["rag_config"] = self.causal_editor_instance.rag_config
                    statistics["rag_retrieval"]["retrieval_enabled"] = self.causal_editor_instance.rag_config.get('use_rag_retrieval', False)
                
                # 从真实统计数据中提取冲突检测信息
                conflict_detector_stats = real_stats.get('conflict_detector_stats', {})
                if conflict_detector_stats:
                    statistics["conflict_detection"].update({
                        "total_detections": conflict_detector_stats.get('detection_count', 0),
                        "conflicts_found": conflict_detector_stats.get('conflict_count', 0),
                        "conflict_rate": conflict_detector_stats.get('conflict_rate', 0.0),
                        "similarity_threshold": conflict_detector_stats.get('similarity_threshold', 0.0),
                        "conflict_threshold": conflict_detector_stats.get('conflict_threshold', 0.0),
                        "layer_conflicts": conflict_detector_stats.get('layer_conflicts', {})
                    })
                
                # 从真实统计数据中提取编辑操作信息
                counterfactual_editor_stats = real_stats.get('counterfactual_editor_stats', {})
                if counterfactual_editor_stats:
                    statistics["editing_operations"].update({
                        "total_edits": counterfactual_editor_stats.get('edit_count', 0),
                        "successful_edits": counterfactual_editor_stats.get('successful_edits', 0),
                        "edit_success_rate": counterfactual_editor_stats.get('success_rate', 0.0),
                        "average_edit_magnitude": counterfactual_editor_stats.get('average_edit_magnitude', 0.0),
                        "edit_strength": counterfactual_editor_stats.get('edit_strength', 0.0),
                        "min_confidence": counterfactual_editor_stats.get('min_confidence', 0.0),
                        "rag_edit_count": counterfactual_editor_stats.get('rag_edit_count', 0),
                        "rag_edit_rate": counterfactual_editor_stats.get('rag_edit_rate', 0.0),
                        "layer_edits": counterfactual_editor_stats.get('layer_edits', {}),
                        "edit_method_stats": counterfactual_editor_stats.get('edit_method_stats', {})
                    })
                
                # 收集动态索引统计信息
                dynamic_index_stats = real_stats.get('dynamic_index_stats', {})
                statistics["dynamic_index_size"] = real_stats.get('dynamic_index_size', 0)
                if dynamic_index_stats:
                    statistics["vector_index"] = {
                        "total_vectors": dynamic_index_stats.get('total_vectors', 0),
                        "dimension": dynamic_index_stats.get('dimension', 0),
                        "index_type": dynamic_index_stats.get('index_type', 'unknown')
                    }
                
                # 收集指纹构建器信息
                if hasattr(self.causal_editor_instance, 'fingerprint_builder'):
                    builder = self.causal_editor_instance.fingerprint_builder
                    if hasattr(builder, 'target_layers'):
                        statistics["fingerprint_builder"] = {
                            "target_layers": builder.target_layers,
                            "fingerprint_dim": getattr(builder, 'fingerprint_dim', 0)
                        }
                
                # 收集动态索引信息
                if hasattr(self.causal_editor_instance, 'dynamic_index'):
                    index = self.causal_editor_instance.dynamic_index
                    statistics["vector_index"] = {
                        "index_type": getattr(index, 'index_type', 'unknown'),
                        "dimension": getattr(index, 'dimension', 0),
                        "max_vectors": getattr(index, 'max_vectors', 0),
                        "current_size": getattr(index, 'current_size', 0) if hasattr(index, 'current_size') else 0
                    }
                
                # 收集当前输入文档信息
                if hasattr(self.causal_editor_instance, 'current_input_doc'):
                    docs = self.causal_editor_instance.current_input_doc
                    statistics["rag_retrieval"]["total_retrievals"] = len(docs) if docs else 0
                    if docs:
                        statistics["rag_retrieval"]["retrieved_documents"] = [
                            {
                                "title": doc.get('title', ''),
                                "score": doc.get('score', 0.0),
                                "source": doc.get('source', ''),
                                "fragment_id": doc.get('fragment_id', '')
                            } for doc in docs[:5]  # 只保存前5个文档的信息
                        ]
                
                # 收集预构建指纹信息
                if hasattr(self.causal_editor_instance, 'prebuilt_fingerprints'):
                    fingerprints = self.causal_editor_instance.prebuilt_fingerprints
                    statistics["fingerprint_cache"] = {
                        "total_fingerprints": len(fingerprints) if fingerprints else 0,
                        "cache_ready": getattr(self.causal_editor_instance, 'prebuilt_index_ready', False)
                    }
                
        except Exception as e:
            statistics["collection_error"] = str(e)
            print(f"⚠️ 收集CausalEditor统计信息时出错: {e}")
        
        return statistics
    
    def _save_causal_editor_analysis(self, statistics: Dict[str, Any], result_path: Path) -> str:
        """
        保存CausalEditor分析报告
        
        Args:
            statistics: CausalEditor统计信息
            result_path: 结果保存路径
            
        Returns:
            保存的文件路径
        """
        analysis_path = result_path / "causal_editor_analysis.json"
        
        # 添加时间戳和元数据
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "device": self.device,
            "causal_editor_config_path": self.causal_editor_config_path,
            "statistics": statistics,
            "analysis_summary": {
                "causal_editor_active": statistics.get("causal_editor_enabled", False),
                "rag_enabled": statistics.get("rag_retrieval", {}).get("retrieval_enabled", False),
                "total_retrieved_docs": statistics.get("rag_retrieval", {}).get("total_retrievals", 0),
                "fingerprint_cache_ready": statistics.get("fingerprint_cache", {}).get("cache_ready", False),
                "vector_index_size": statistics.get("vector_index", {}).get("current_size", 0)
            }
        }
        
        # 保存分析数据
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ CausalEditor分析报告已保存: {analysis_path}")
        return str(analysis_path)
    
    def save_results(self, dataset, result_dir: str):
        """
        保存结果到指定目录
        
        Args:
            dataset: 处理后的数据集
            result_dir: 结果保存目录
        """
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        
        # 准备结果数据
        results = []
        for i, item in enumerate(dataset):
            result_entry = {
                "question_id": i + 1,
                "question": item.question,
                "generated_answer": getattr(item, 'pred', ''),
                "golden_answers": getattr(item, 'golden_answers', ''),
                "generation_time": getattr(item, 'generation_time', 0.0),
                "success": getattr(item, 'success', False),
                "timestamp": datetime.now().isoformat()
            }
            
            # 添加错误信息（如果有）
            try:
                result_entry["error"] = item.error
            except (AttributeError, KeyError):
                pass
            
            # 添加其他属性（如果存在）
            for attr in ['category', 'type', 'best_answer', 'correct_answers', 'incorrect_answers']:
                try:
                    result_entry[attr] = getattr(item, attr)
                except (AttributeError, KeyError):
                    pass
            
            results.append(result_entry)
        
        # 保存干净的结果文件
        clean_results_path = result_path / "causal_editor_clean_results.json"
        clean_results = []
        for result in results:
            if result.get('success', False):
                clean_entry = {
                    "question_id": result['question_id'],
                    "question": result['question'],
                    "generated_answer": result['generated_answer'],
                    "golden_answers": result['golden_answers']
                }
                if 'category' in result:
                    clean_entry['category'] = result['category']
                if 'type' in result:
                    clean_entry['type'] = result['type']
                clean_results.append(clean_entry)
        
        with open(clean_results_path, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 干净结果文件已保存: {clean_results_path}")
        
        # 保存详细结果文件
        debug_results_path = result_path / "causal_editor_debug_results.json"
        
        # 计算统计信息
        successful_tests = [r for r in results if r.get('success', False)]
        failed_tests = [r for r in results if not r.get('success', False)]
        total_generation_time = sum(r.get('generation_time', 0) for r in successful_tests)
        avg_generation_time = total_generation_time / len(successful_tests) if successful_tests else 0
        
        debug_summary = {
            "model_name": self.model_name,
            "test_type": "causal_editor_pipeline",
            "device": self.device,
            "causal_editor_config_path": self.causal_editor_config_path,
            "test_time": datetime.now().isoformat(),
            "total_questions": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(results) * 100 if results else 0,
            "avg_generation_time": avg_generation_time,
            "detailed_results": results
        }

        with open(debug_results_path, "w", encoding="utf-8") as f:
            json.dump(debug_summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 调试信息文件已保存: {debug_results_path}")
        
        # 保存统计报告
        stats_report_path = result_path / "causal_editor_statistics_report.txt"
        
        with open(stats_report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CausalEditor Pipeline 统计报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"🤖 模型: {self.model_name}\n")
            f.write(f"💻 设备: {self.device}\n")
            f.write(f"🔍 CausalEditor配置: {self.causal_editor_config_path}\n")
            f.write(f"⏰ 测试时间: {datetime.now()}\n")
            f.write(f"📊 问题总数: {len(results)}\n")
            f.write(f"✅ 成功数量: {len(successful_tests)}\n")
            f.write(f"❌ 失败数量: {len(failed_tests)}\n")
            f.write(f"✅ 成功率: {len(successful_tests) / len(results) * 100:.1f}%\n")
            f.write(f"⚡ 平均生成时间: {avg_generation_time:.3f}s\n")
            f.write(f"📁 结果文件:\n")
            f.write(f"  - 干净结果: {clean_results_path}\n")
            f.write(f"  - 调试信息: {debug_results_path}\n")
            f.write(f"  - 统计报告: {stats_report_path}\n")
            
            # 添加CausalEditor分析信息
            try:
                causal_editor_statistics = self._collect_causal_editor_statistics()
                f.write(f"\n🧠 CausalEditor 分析:\n")
                f.write(f"  - 启用状态: {'是' if causal_editor_statistics.get('causal_editor_enabled', False) else '否'}\n")
                f.write(f"  - RAG检索: {'启用' if causal_editor_statistics.get('rag_retrieval', {}).get('retrieval_enabled', False) else '禁用'}\n")
                f.write(f"  - 检索文档数: {causal_editor_statistics.get('rag_retrieval', {}).get('total_retrievals', 0)}\n")
                f.write(f"  - 指纹缓存: {'就绪' if causal_editor_statistics.get('fingerprint_cache', {}).get('cache_ready', False) else '未就绪'}\n")
                f.write(f"  - 向量索引大小: {causal_editor_statistics.get('vector_index', {}).get('current_size', 0)}\n")
                
                # 添加配置信息
                conflict_detection = causal_editor_statistics.get('conflict_detection', {})
                if 'similarity_threshold' in conflict_detection:
                    f.write(f"  - 相似度阈值: {conflict_detection['similarity_threshold']}\n")
                if 'conflict_threshold' in conflict_detection:
                    f.write(f"  - 冲突阈值: {conflict_detection['conflict_threshold']}\n")
                    
                editing_ops = causal_editor_statistics.get('editing_operations', {})
                if 'edit_strength' in editing_ops:
                    f.write(f"  - 编辑强度: {editing_ops['edit_strength']}\n")
                    
            except Exception as e:
                f.write(f"\n⚠️ CausalEditor分析收集失败: {e}\n")
        
        print(f"✅ 统计报告已保存: {stats_report_path}")
        
        # 收集并保存CausalEditor统计信息
        causal_editor_analysis_path = None
        try:
            causal_editor_statistics = self._collect_causal_editor_statistics()
            causal_editor_analysis_path = self._save_causal_editor_analysis(causal_editor_statistics, result_path)
        except Exception as e:
            print(f"⚠️ 保存CausalEditor分析报告时出错: {e}")
        
        result_files = {
            "clean_results_path": clean_results_path,
            "debug_results_path": debug_results_path,
            "stats_report_path": stats_report_path
        }
        
        if causal_editor_analysis_path:
            result_files["causal_editor_analysis_path"] = causal_editor_analysis_path
        
        return result_files



if __name__ == "__main__":
    # 示例使用
    print("CausalEditor Pipeline 示例")
    
    # 配置参数
    MODEL_NAME = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/Llama-3.1-8B-Instruct"
    RETRIEVAL_CONFIG_PATH = "./configs/retrieval_config.yaml"
    RESULT_DIR = "./result/result_causal_editor_pipeline"
    
    try:
        # 加载FlashRAG配置
        from flashrag.config import Config
        config = Config(config_file_path=RETRIEVAL_CONFIG_PATH)
        
        
        # 创建测试数据集
        from flashrag.dataset import Dataset
        
        # 方法1: 使用自定义数据
        test_data = [
            {
                "id": "1",
                "question": "What is the capital of France?",
                "golden_answers": ["Paris"]
            },
            {
                "id": "2", 
                "question": "Who wrote Romeo and Juliet?",
                "golden_answers": ["William Shakespeare", "Shakespeare"]
            },
            {
                "id": "3",
                "question": "What is the largest planet in our solar system?",
                "golden_answers": ["Jupiter"]
            },
            {
                "id": "4",
                "question": "What is the capital of Japan?",
                "golden_answers": ["Tokyo"]
            }
        ]
        
        dataset = Dataset(config=config, data=test_data)
        print(f"✅ 成功创建测试数据集，包含 {len(dataset)} 个问题")

        # 初始化pipeline
        pipeline = CausalEditorPipeline(
            config=config,
            model_name=MODEL_NAME,
            causal_editor_config_path="./configs/causal_editor.json",
            max_length=4096,
            max_new_tokens=256
        )
        
        # 运行pipeline
        result_dataset = pipeline.run(dataset, do_eval=True)
        
        # 保存结果
        saved_files = pipeline.save_results(result_dataset, RESULT_DIR)
        
        print("\n" + "=" * 80)
        print("CausalEditor Pipeline 测试完成！")
        print("生成的文件:")
        for file_type, file_path in saved_files.items():
            print(f"  - {file_type}: {file_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()