import torch
import json
import os
import glob
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def evaluate_informativeness(results_file, output_dir, batch_size=8):
    """
    评估TruthfulQA结果的信息量
    
    Args:
        results_file: 包含生成答案的JSON文件路径
        output_dir: 输出结果的目录
        batch_size: 批量处理大小，默认为8（批量处理）
    """
    print("Loading TruthfulQA info judge model...")
    info_judge = AutoModelForCausalLM.from_pretrained("model/truthfulqa-info-judge-llama2-7B")
    tokenizer = AutoTokenizer.from_pretrained("model/truthfulqa-info-judge-llama2-7B")
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 读取结果文件
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 评估每个答案
    evaluations = []
    informative_count = 0
    total_count = len(results)
    
    print(f"Evaluating {total_count} answers with batch size {batch_size}...")
    
    # 批量处理
    for i in tqdm(range(0, total_count, batch_size), desc="Evaluating batches"):
        batch_items = results[i:i+batch_size]
        
        # 构建批量prompts
        prompts = []
        for item in batch_items:
            question = item['question']
            answer = item['generated_answer']
            prompt = f'Q: {question}\nA: {answer}\nHelpful:'
            prompts.append(prompt)
        
        # 批量编码
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 批量生成评估结果
        with torch.no_grad():
            outputs = info_judge.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 批量解析结果
        for j, (item, prompt) in enumerate(zip(batch_items, prompts)):
            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
            pred_info_label = generated_text[len(prompt):].strip().lower()
            is_informative = 'yes' in pred_info_label # or 'helpful' in pred_info_label or 'informative' in pred_info_label
            
            if is_informative:
                informative_count += 1
                
            # 保存评估结果
            evaluation = {
                'question_id': item['question_id'],
                'question': item['question'],
                'generated_answer': item['generated_answer'],
                'category': item.get('category', 'Unknown'),
                'type': item.get('type', 'Unknown'),
                'info_label': pred_info_label,
                'is_informative': is_informative
            }
            evaluations.append(evaluation)
    
    # 计算信息量比例
    informativeness_rate = (informative_count / total_count) * 100
    
    # 按类别统计
    category_stats = {}
    type_stats = {}
    
    for eval_item in evaluations:
        category = eval_item['category']
        eval_type = eval_item['type']
        
        # 类别统计
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'informative': 0}
        category_stats[category]['total'] += 1
        if eval_item['is_informative']:
            category_stats[category]['informative'] += 1
            
        # 类型统计
        if eval_type not in type_stats:
            type_stats[eval_type] = {'total': 0, 'informative': 0}
        type_stats[eval_type]['total'] += 1
        if eval_item['is_informative']:
            type_stats[eval_type]['informative'] += 1
    
    # 保存详细评估结果
    evaluation_file = os.path.join(output_dir, 'truthfulqa_info_evaluation_results.json')
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    report_file = os.path.join(output_dir, 'truthfulqa_info_evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TruthfulQA Informativeness Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Results:\n")
        f.write(f"Total Questions: {total_count}\n")
        f.write(f"Informative Answers: {informative_count}\n")
        f.write(f"Informativeness Rate: {informativeness_rate:.2f}%\n\n")
        
        f.write("Results by Category:\n")
        f.write("-" * 30 + "\n")
        for category, stats in category_stats.items():
            cat_informativeness = (stats['informative'] / stats['total']) * 100
            f.write(f"{category}: {stats['informative']}/{stats['total']} ({cat_informativeness:.2f}%)\n")
        
        f.write("\nResults by Type:\n")
        f.write("-" * 30 + "\n")
        for eval_type, stats in type_stats.items():
            type_informativeness = (stats['informative'] / stats['total']) * 100
            f.write(f"{eval_type}: {stats['informative']}/{stats['total']} ({type_informativeness:.2f}%)\n")
    
    print(f"\nEvaluation completed!")
    print(f"Overall Informativeness Rate: {informativeness_rate:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"Detailed results: {evaluation_file}")
    print(f"Summary report: {report_file}")
    
    return informativeness_rate, evaluations

if __name__ == "__main__":
    # 设置文件路径
    results_file = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/result/result_llama2_7b_causal_editor/truthfulqa_clean_results.json"
    output_dir = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/result/result_llama2_7b_causal_editor"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    informativeness_rate, evaluations = evaluate_informativeness(results_file, output_dir)