import csv
import random
import json
from collections import defaultdict
from pathlib import Path

# 设置固定随机种子确保结果可重现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 配置参数
TRUTHFULQA_DATA_PATH = "TruthfulQA/data/v1/TruthfulQA.csv"
SAMPLES_PER_CATEGORY = 1
OUTPUT_PATH = "./TruthfulQA/TruthfulQA_sampled_38.csv"

def load_and_sample_data():
    """加载TruthfulQA数据并按类别采样"""
    
    # 按类别分组数据
    category_data = defaultdict(list)
    
    print("正在加载TruthfulQA数据...")
    with open(TRUTHFULQA_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['Category']
            category_data[category].append(row)
    
    print(f"总共找到 {len(category_data)} 个类别")
    
    # 显示每个类别的数据量
    for category, items in category_data.items():
        print(f"  {category}: {len(items)} 个问题")
    
    # 从每个类别中采样
    sampled_data = []
    sampling_stats = {}
    
    print(f"\n开始采样，每个类别采样 {SAMPLES_PER_CATEGORY} 个问题...")
    
    for category, items in category_data.items():
        if len(items) >= SAMPLES_PER_CATEGORY:
            # 如果数据足够，随机采样
            sampled_items = random.sample(items, SAMPLES_PER_CATEGORY)
            sampling_stats[category] = SAMPLES_PER_CATEGORY
        else:
            # 如果数据不足，取全部
            sampled_items = items
            sampling_stats[category] = len(items)
            print(f"  警告: {category} 类别只有 {len(items)} 个问题，少于目标 {SAMPLES_PER_CATEGORY} 个")
        
        sampled_data.extend(sampled_items)
    
    print(f"\n采样完成，总共采样了 {len(sampled_data)} 个问题")
    
    # 显示采样统计
    print("\n采样统计:")
    total_sampled = 0
    for category, count in sampling_stats.items():
        print(f"  {category}: {count} 个问题")
        total_sampled += count
    
    print(f"\n总计: {total_sampled} 个问题")
    
    return sampled_data, sampling_stats

def save_sampled_data(sampled_data, output_path):
    """保存采样后的数据到CSV文件"""
    
    print(f"\n正在保存采样数据到 {output_path}...")
    
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入CSV文件
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if sampled_data:
            fieldnames = sampled_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_data)
    
    print(f"✅ 采样数据已保存到 {output_path}")

def save_sampling_report(sampling_stats, output_dir):
    """保存采样报告"""
    
    report_path = Path(output_dir) / "sampling_report.json"
    
    report = {
        "random_seed": RANDOM_SEED,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "total_categories": len(sampling_stats),
        "total_sampled_questions": sum(sampling_stats.values()),
        "category_stats": sampling_stats,
        "categories_with_insufficient_data": [
            category for category, count in sampling_stats.items() 
            if count < SAMPLES_PER_CATEGORY
        ]
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 采样报告已保存到 {report_path}")

def main():
    print("=" * 80)
    print("TruthfulQA 数据集采样工具")
    print("=" * 80)
    print(f"随机种子: {RANDOM_SEED}")
    print(f"每类别采样数: {SAMPLES_PER_CATEGORY}")
    print(f"输入文件: {TRUTHFULQA_DATA_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("=" * 80)
    
    try:
        # 加载和采样数据
        sampled_data, sampling_stats = load_and_sample_data()
        
        # 保存采样数据
        save_sampled_data(sampled_data, OUTPUT_PATH)
        
        # 保存采样报告
        output_dir = Path(OUTPUT_PATH).parent
        save_sampling_report(sampling_stats, output_dir)
        
        print("\n" + "=" * 80)
        print("采样完成！")
        print(f"采样数据文件: {OUTPUT_PATH}")
        print(f"采样报告文件: {output_dir}/sampling_report.json")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 采样过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()