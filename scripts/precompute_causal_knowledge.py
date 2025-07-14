#!/usr/bin/env python3
"""
CausalEditor 预计算脚本
用于执行知识提取和激活指纹构建的完整流程

使用示例:
python scripts/precompute_causal_knowledge.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-dir ./precomputed_data/llama2-7b \
    --knowledge-type wikidata \
    --limit 10000 \
    --device cuda \
    --batch-size 8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from causal_editor.precompute.precompute_pipeline import PrecomputePipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CausalEditor 预计算脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 从Wikidata提取10000个三元组并构建激活指纹:
   python scripts/precompute_causal_knowledge.py \\
       --model-name meta-llama/Llama-2-7b-hf \\
       --output-dir ./precomputed_data/llama2-7b \\
       --knowledge-type wikidata \\
       --limit 10000

2. 从CSV文件构建激活指纹:
   python scripts/precompute_causal_knowledge.py \\
       --model-name mistralai/Mistral-7B-v0.1 \\
       --output-dir ./precomputed_data/mistral-7b \\
       --knowledge-type csv \\
       --csv-path ./data/knowledge_triplets.csv \\
       --limit 5000

3. 只在特定层构建指纹:
   python scripts/precompute_causal_knowledge.py \\
       --model-name meta-llama/Llama-2-7b-hf \\
       --target-layers 10 11 12 13 14 15 \\
       --limit 10000
        """
    )
    
    # 模型参数
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称或路径 (例如: meta-llama/Llama-2-7b-hf)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./precomputed_data",
        help="输出目录 (默认: ./precomputed_data)"
    )
    
    # 知识提取参数
    parser.add_argument(
        "--knowledge-type",
        type=str,
        choices=["wikidata", "csv"],
        default="wikidata",
        help="知识源类型 (默认: wikidata)"
    )
    
    parser.add_argument(
        "--csv-path",
        type=str,
        help="CSV文件路径 (当knowledge-type为csv时必需)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="提取的三元组数量限制 (默认: 10000)"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="API请求速率限制，秒 (默认: 1.0)"
    )
    
    # 模型参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (默认: cuda)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批处理大小 (默认: 8)"
    )
    
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        help="目标层列表 (例如: --target-layers 10 11 12). 不指定则使用所有层"
    )
    
    # 功能选项
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量更新现有数据库"
    )
    
    parser.add_argument(
        "--existing-db-path",
        type=str,
        help="现有数据库路径 (增量模式必需)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证现有数据库"
    )
    
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="完成后清理缓存"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="配置文件路径 (JSON格式)"
    )
    
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="生成配置模板文件"
    )
    
    # 日志参数
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_config(config_file: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        sys.exit(1)


def generate_config_template(output_path: str):
    """生成配置模板"""
    pipeline = PrecomputePipeline("dummy", "dummy")
    template = pipeline.get_config_template()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"配置模板已生成: {output_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 生成配置模板
    if args.generate_config:
        config_path = args.config_file or "causal_editor_config.json"
        generate_config_template(config_path)
        return
    
    # 验证参数
    if args.knowledge_type == "csv" and not args.csv_path:
        logging.error("使用CSV知识源时必须指定 --csv-path")
        sys.exit(1)
    
    if args.incremental and not args.existing_db_path:
        logging.error("增量模式需要指定 --existing-db-path")
        sys.exit(1)
    
    # 从配置文件加载参数
    if args.config_file:
        config = load_config(args.config_file)
        # 命令行参数会覆盖配置文件
        model_name = args.model_name or config.get("model", {}).get("name")
        device = args.device or config.get("model", {}).get("device", "cuda")
        batch_size = args.batch_size or config.get("model", {}).get("batch_size", 8)
        target_layers = args.target_layers or config.get("fingerprint", {}).get("target_layers")
        limit = args.limit or config.get("limits", {}).get("max_triplets", 10000)
    else:
        model_name = args.model_name
        device = args.device
        batch_size = args.batch_size
        target_layers = args.target_layers
        limit = args.limit
    
    # 初始化流水线
    pipeline = PrecomputePipeline(
        model_name=model_name,
        output_dir=args.output_dir,
        device=device,
        batch_size=batch_size
    )
    
    try:
        # 仅验证模式
        if args.validate_only:
            if not args.existing_db_path:
                logging.error("验证模式需要指定数据库路径")
                sys.exit(1)
            
            logging.info("开始验证数据库...")
            result = pipeline.validate_database(args.existing_db_path)
            
            print("\n验证结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if result['status'] == 'success':
                logging.info("数据库验证通过")
            else:
                logging.warning(f"数据库验证失败: {result.get('message', 'Unknown error')}")
            
            return
        
        # 准备知识提取配置
        knowledge_config = {
            'type': args.knowledge_type,
            'rate_limit': args.rate_limit
        }
        
        if args.knowledge_type == 'csv':
            knowledge_config['csv_path'] = args.csv_path
        
        # 运行流水线
        if args.incremental:
            logging.info("开始增量更新...")
            db_path = pipeline.run_incremental_update(
                new_knowledge_config=knowledge_config,
                existing_db_path=args.existing_db_path,
                limit=limit
            )
        else:
            logging.info("开始完整预计算流水线...")
            db_path = pipeline.run_full_pipeline(
                knowledge_config=knowledge_config,
                limit=limit,
                target_layers=target_layers
            )
        
        # 验证结果
        logging.info("验证生成的数据库...")
        validation_result = pipeline.validate_database(db_path)
        
        if validation_result['status'] == 'success':
            logging.info("数据库构建并验证成功!")
            print(f"\n✅ 预计算完成!")
            print(f"📊 向量数据库路径: {db_path}")
            print(f"📈 总向量数: {validation_result.get('total_vectors', 0)}")
            print(f"📐 向量维度: {validation_result.get('dimension', 0)}")
        else:
            logging.error(f"数据库验证失败: {validation_result.get('message', 'Unknown error')}")
            sys.exit(1)
        
        # 清理缓存
        if args.cleanup_cache:
            logging.info("清理缓存...")
            pipeline.cleanup_cache()
        
        logging.info("预计算流水线执行完成!")
        
    except KeyboardInterrupt:
        logging.info("用户中断执行")
        sys.exit(1)
    except Exception as e:
        logging.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 