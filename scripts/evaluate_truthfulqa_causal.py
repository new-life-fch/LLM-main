#!/usr/bin/env python3
"""
CausalEditor TruthfulQA评估脚本
评估CausalEditor在TruthfulQA数据集上的表现，并与baseline方法对比

使用示例:
python scripts/evaluate_truthfulqa_causal.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --vector-db-path ./precomputed_data/llama2-7b/vector_database \
    --output-dir ./results/truthfulqa_causal \
    --edit-strength 1.0 \
    --top-layers 10 \
    --mode generation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入原始TruthfulQA评估工具
try:
    from TruthfulQA.truthfulqa.utilities import (
        format_prompt, format_prompt_with_answer_strings,
        split_multi_answer, format_best, find_start
    )
    from TruthfulQA.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL
    from TruthfulQA.truthfulqa.models import set_columns, MC_calcs
    from TruthfulQA.truthfulqa.evaluate import format_frame
except ImportError:
    logging.warning("无法导入TruthfulQA模块，某些功能可能不可用")

from modeling_llms.modeling_llama_causal import CausalLlamaForCausalLM
from transformers import AutoTokenizer
import torch


class CausalTruthfulQAEvaluator:
    """
    CausalEditor TruthfulQA评估器
    """
    
    def __init__(
        self,
        model_path: str,
        vector_db_path: str,
        edit_strength: float = 1.0,
        top_layers: int = 10,
        similarity_threshold: float = 0.8,
        conflict_threshold: float = 0.6,
        device: str = "cuda"
    ):
        """
        初始化评估器
        
        Args:
            model_path: 基座模型路径
            vector_db_path: 预计算的向量数据库路径
            edit_strength: 编辑强度
            top_layers: 参与编辑的层数
            similarity_threshold: 相似度阈值
            conflict_threshold: 冲突阈值
            device: 计算设备
        """
        self.model_path = model_path
        self.vector_db_path = vector_db_path
        self.device = device
        
        # 加载模型和tokenizer
        logging.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载CausalLlama模型
        self.model = CausalLlamaForCausalLM.from_pretrained_with_causal_editor(
            model_name_or_path=model_path,
            vector_db_path=vector_db_path,
            edit_strength=edit_strength,
            top_layers=top_layers,
            similarity_threshold=similarity_threshold,
            conflict_threshold=conflict_threshold,
            device=device,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.set_tokenizer(self.tokenizer)
        self.model.eval()
        
        # 加载baseline模型用于对比
        self.baseline_model = None
        
        logging.info("CausalTruthfulQA评估器初始化完成")
    
    def load_baseline_model(self):
        """加载baseline模型用于对比"""
        if self.baseline_model is None:
            logging.info("加载baseline模型...")
            from transformers import AutoModelForCausalLM
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.baseline_model.eval()
    
    def evaluate_multiple_choice(
        self,
        questions_file: str = "TruthfulQA/data/TruthfulQA.csv",
        output_path: str = "./results/mc_results.json",
        compare_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        评估多选题任务
        
        Args:
            questions_file: 问题文件路径
            output_path: 输出路径
            compare_baseline: 是否与baseline对比
            
        Returns:
            评估结果
        """
        logging.info("开始多选题评估...")
        
        # 加载问题
        questions_df = pd.read_csv(questions_file)
        questions_df.dropna(axis=1, how="all", inplace=True)
        
        results = {
            'causal_editor': [],
            'baseline': [] if compare_baseline else None,
            'comparisons': []
        }
        
        # 设置MC模式
        self.model.set_generation_mode(is_mc=True)
        
        if compare_baseline:
            self.load_baseline_model()
        
        for idx, row in questions_df.iterrows():
            if pd.isnull(row[INCORRECT_COL]):
                continue
            
            question = row["Question"]
            ref_true = split_multi_answer(row[ANSWER_COL])
            ref_false = split_multi_answer(row[INCORRECT_COL])
            
            logging.info(f"评估问题 {idx + 1}/{len(questions_df)}: {question[:50]}...")
            
            # CausalEditor评估
            causal_result = self._evaluate_mc_question(
                question, ref_true, ref_false, use_causal=True
            )
            results['causal_editor'].append(causal_result)
            
            # Baseline评估
            if compare_baseline:
                baseline_result = self._evaluate_mc_question(
                    question, ref_true, ref_false, use_causal=False
                )
                results['baseline'].append(baseline_result)
                
                # 对比分析
                comparison = self._compare_mc_results(causal_result, baseline_result)
                results['comparisons'].append(comparison)
        
        # 计算汇总统计
        summary = self._compute_mc_summary(results)
        results['summary'] = summary
        
        # 保存结果
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"多选题评估完成，结果保存到: {output_path}")
        return results
    
    def _evaluate_mc_question(
        self, 
        question: str, 
        ref_true: List[str], 
        ref_false: List[str],
        use_causal: bool = True
    ) -> Dict[str, Any]:
        """
        评估单个多选题
        
        Args:
            question: 问题
            ref_true: 正确答案列表
            ref_false: 错误答案列表
            use_causal: 是否使用CausalEditor
            
        Returns:
            评估结果
        """
        model_to_use = self.model if use_causal else self.baseline_model
        
        # 设置提示长度
        prompt = format_prompt(question)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        if use_causal:
            self.model.set_generation_mode(is_mc=True, prompt_length=prompt_length)
        
        scores_true = []
        scores_false = []
        
        # 评估正确答案
        for answer in ref_true:
            score = self._get_answer_log_prob(model_to_use, question, answer)
            scores_true.append(score)
        
        # 评估错误答案
        for answer in ref_false:
            score = self._get_answer_log_prob(model_to_use, question, answer)
            scores_false.append(score)
        
        # 计算准确率
        mc1_acc = self._compute_mc1_accuracy(scores_true, scores_false)
        mc2_acc = self._compute_mc2_accuracy(scores_true, scores_false)
        
        return {
            'question': question,
            'scores_true': scores_true,
            'scores_false': scores_false,
            'mc1_accuracy': mc1_acc,
            'mc2_accuracy': mc2_acc,
            'use_causal': use_causal
        }
    
    def _get_answer_log_prob(self, model, question: str, answer: str) -> float:
        """
        获取答案的对数概率
        
        Args:
            model: 使用的模型
            question: 问题
            answer: 答案
            
        Returns:
            对数概率
        """
        # 构建完整输入
        full_text = f"{question} {answer}"
        
        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        question_inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 获取logits
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 计算答案部分的对数概率
            question_len = question_inputs['input_ids'].shape[1]
            answer_logits = logits[0, question_len-1:-1]  # 答案部分的logits
            answer_tokens = inputs['input_ids'][0, question_len:]  # 答案tokens
            
            # 计算对数概率
            log_probs = F.log_softmax(answer_logits, dim=-1)
            answer_log_prob = 0.0
            
            for i, token_id in enumerate(answer_tokens):
                if i < len(log_probs):
                    answer_log_prob += log_probs[i, token_id].item()
        
        return answer_log_prob
    
    def _compute_mc1_accuracy(self, scores_true: List[float], scores_false: List[float]) -> float:
        """计算MC1准确率"""
        if not scores_true or not scores_false:
            return 0.0
        
        max_true = max(scores_true)
        max_false = max(scores_false)
        
        return 1.0 if max_true > max_false else 0.0
    
    def _compute_mc2_accuracy(self, scores_true: List[float], scores_false: List[float]) -> float:
        """计算MC2准确率"""
        if not scores_true or not scores_false:
            return 0.0
        
        # 归一化概率
        all_scores = scores_true + scores_false
        exp_scores = [np.exp(score) for score in all_scores]
        sum_exp = sum(exp_scores)
        
        # 正确答案的概率质量
        true_prob_mass = sum(exp_scores[:len(scores_true)]) / sum_exp
        
        return true_prob_mass
    
    def _compare_mc_results(
        self, 
        causal_result: Dict[str, Any], 
        baseline_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对比CausalEditor和baseline的结果"""
        return {
            'question': causal_result['question'],
            'causal_mc1': causal_result['mc1_accuracy'],
            'baseline_mc1': baseline_result['mc1_accuracy'],
            'causal_mc2': causal_result['mc2_accuracy'],
            'baseline_mc2': baseline_result['mc2_accuracy'],
            'mc1_improvement': causal_result['mc1_accuracy'] - baseline_result['mc1_accuracy'],
            'mc2_improvement': causal_result['mc2_accuracy'] - baseline_result['mc2_accuracy']
        }
    
    def _compute_mc_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算多选题汇总统计"""
        causal_results = results['causal_editor']
        
        summary = {
            'causal_editor': {
                'mc1_accuracy': np.mean([r['mc1_accuracy'] for r in causal_results]),
                'mc2_accuracy': np.mean([r['mc2_accuracy'] for r in causal_results]),
                'total_questions': len(causal_results)
            }
        }
        
        if results['baseline']:
            baseline_results = results['baseline']
            summary['baseline'] = {
                'mc1_accuracy': np.mean([r['mc1_accuracy'] for r in baseline_results]),
                'mc2_accuracy': np.mean([r['mc2_accuracy'] for r in baseline_results]),
                'total_questions': len(baseline_results)
            }
            
            # 改进统计
            comparisons = results['comparisons']
            summary['improvements'] = {
                'mc1_improvement': np.mean([c['mc1_improvement'] for c in comparisons]),
                'mc2_improvement': np.mean([c['mc2_improvement'] for c in comparisons]),
                'mc1_improvement_std': np.std([c['mc1_improvement'] for c in comparisons]),
                'mc2_improvement_std': np.std([c['mc2_improvement'] for c in comparisons])
            }
        
        return summary
    
    def evaluate_generation(
        self,
        questions_file: str = "TruthfulQA/data/TruthfulQA.csv",
        output_path: str = "./results/generation_results.jsonl",
        max_new_tokens: int = 100,
        compare_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        评估开放式生成任务
        
        Args:
            questions_file: 问题文件路径
            output_path: 输出路径
            max_new_tokens: 最大生成token数
            compare_baseline: 是否与baseline对比
            
        Returns:
            评估结果
        """
        logging.info("开始开放式生成评估...")
        
        # 加载问题
        questions_df = pd.read_csv(questions_file)
        questions_df.dropna(axis=1, how="all", inplace=True)
        
        # 设置生成模式
        self.model.set_generation_mode(is_mc=False)
        
        if compare_baseline:
            self.load_baseline_model()
        
        results = []
        
        for idx, row in questions_df.iterrows():
            question = row["Question"]
            
            logging.info(f"生成问题 {idx + 1}/{len(questions_df)}: {question[:50]}...")
            
            # CausalEditor生成
            causal_answer = self._generate_answer(
                question, max_new_tokens, use_causal=True
            )
            
            result = {
                'question_id': idx,
                'question': question,
                'causal_answer': causal_answer,
                'reference_answers': row.get(ANSWER_COL, ''),
                'reference_incorrect': row.get(INCORRECT_COL, '')
            }
            
            # Baseline生成
            if compare_baseline:
                baseline_answer = self._generate_answer(
                    question, max_new_tokens, use_causal=False
                )
                result['baseline_answer'] = baseline_answer
            
            results.append(result)
        
        # 保存结果
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        logging.info(f"开放式生成评估完成，结果保存到: {output_path}")
        return {'results': results, 'total_questions': len(results)}
    
    def _generate_answer(
        self, 
        question: str, 
        max_new_tokens: int, 
        use_causal: bool = True
    ) -> str:
        """
        生成问题答案
        
        Args:
            question: 问题
            max_new_tokens: 最大生成token数
            use_causal: 是否使用CausalEditor
            
        Returns:
            生成的答案
        """
        model_to_use = self.model if use_causal else self.baseline_model
        
        # Tokenize输入
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 生成
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                is_mc_mode=False if use_causal else None
            )
            
            # 解码答案（排除输入部分）
            answer_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def get_causal_editor_statistics(self) -> Dict[str, Any]:
        """获取CausalEditor运行统计"""
        return self.model.get_causal_editor_statistics()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CausalEditor TruthfulQA评估")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="基座模型路径")
    parser.add_argument("--vector-db-path", type=str, required=True,
                       help="预计算的向量数据库路径")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="输出目录")
    
    # CausalEditor参数
    parser.add_argument("--edit-strength", type=float, default=1.0,
                       help="编辑强度")
    parser.add_argument("--top-layers", type=int, default=10,
                       help="参与编辑的层数")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                       help="相似度阈值")
    parser.add_argument("--conflict-threshold", type=float, default=0.6,
                       help="冲突检测阈值")
    
    # 评估参数
    parser.add_argument("--mode", type=str, choices=["mc", "generation", "both"],
                       default="both", help="评估模式")
    parser.add_argument("--questions-file", type=str,
                       default="TruthfulQA/data/TruthfulQA.csv",
                       help="问题文件路径")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="生成任务的最大token数")
    parser.add_argument("--no-baseline", action="store_true",
                       help="不与baseline对比")
    
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化评估器
        evaluator = CausalTruthfulQAEvaluator(
            model_path=args.model_path,
            vector_db_path=args.vector_db_path,
            edit_strength=args.edit_strength,
            top_layers=args.top_layers,
            similarity_threshold=args.similarity_threshold,
            conflict_threshold=args.conflict_threshold,
            device=args.device
        )
        
        results = {}
        
        # 多选题评估
        if args.mode in ["mc", "both"]:
            mc_results = evaluator.evaluate_multiple_choice(
                questions_file=args.questions_file,
                output_path=str(output_dir / "mc_results.json"),
                compare_baseline=not args.no_baseline
            )
            results['multiple_choice'] = mc_results['summary']
        
        # 生成任务评估
        if args.mode in ["generation", "both"]:
            gen_results = evaluator.evaluate_generation(
                questions_file=args.questions_file,
                output_path=str(output_dir / "generation_results.jsonl"),
                max_new_tokens=args.max_new_tokens,
                compare_baseline=not args.no_baseline
            )
            results['generation'] = {
                'total_questions': gen_results['total_questions']
            }
        
        # 获取CausalEditor统计
        causal_stats = evaluator.get_causal_editor_statistics()
        results['causal_editor_stats'] = causal_stats
        
        # 保存综合结果
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print("\n" + "="*50)
        print("CausalEditor TruthfulQA评估结果")
        print("="*50)
        
        if 'multiple_choice' in results:
            mc_res = results['multiple_choice']
            print(f"\n多选题结果:")
            if 'causal_editor' in mc_res:
                print(f"  CausalEditor MC1: {mc_res['causal_editor']['mc1_accuracy']:.3f}")
                print(f"  CausalEditor MC2: {mc_res['causal_editor']['mc2_accuracy']:.3f}")
            
            if 'baseline' in mc_res:
                print(f"  Baseline MC1: {mc_res['baseline']['mc1_accuracy']:.3f}")
                print(f"  Baseline MC2: {mc_res['baseline']['mc2_accuracy']:.3f}")
            
            if 'improvements' in mc_res:
                imp = mc_res['improvements']
                print(f"  MC1改进: {imp['mc1_improvement']:+.3f}")
                print(f"  MC2改进: {imp['mc2_improvement']:+.3f}")
        
        if 'generation' in results:
            print(f"\n生成任务:")
            print(f"  总问题数: {results['generation']['total_questions']}")
        
        # CausalEditor统计
        if causal_stats:
            print(f"\nCausalEditor统计:")
            if 'conflict_detector_stats' in causal_stats:
                cd_stats = causal_stats['conflict_detector_stats']
                print(f"  冲突检测次数: {cd_stats.get('detection_count', 0)}")
                print(f"  冲突发现次数: {cd_stats.get('conflict_count', 0)}")
                if cd_stats.get('detection_count', 0) > 0:
                    print(f"  冲突率: {cd_stats.get('conflict_rate', 0):.3f}")
            
            if 'counterfactual_editor_stats' in causal_stats:
                ce_stats = causal_stats['counterfactual_editor_stats']
                print(f"  编辑次数: {ce_stats.get('edit_count', 0)}")
                print(f"  成功率: {ce_stats.get('success_rate', 0):.3f}")
        
        print(f"\n结果已保存到: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 