#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建RAG索引脚本

该脚本用于从flashrag_corpus.jsonl文件构建FAISS向量索引，
支持GPU加速和IVF+PQ索引类型。

使用方法:
    python build_index.py [--corpus_path PATH] [--model_name MODEL] [--batch_size SIZE]
"""

import argparse
import logging
import sys
import time
import torch
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from causal_editor.dynamic.rag_retriever import RAGRetriever
from causal_editor.utils.path_config import PathConfig


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('build_index.log', encoding='utf-8')
        ]
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='构建RAG FAISS索引',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--corpus_path',
        type=str,
        default='wiki_data/flashrag_corpus.jsonl',
        help='FlashRAG语料库文件路径'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='/root/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09',
        help='嵌入模型名称'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='批处理大小'
    )
    
    parser.add_argument(
        '--max_docs',
        type=int,
        default=None,
        help='最大处理文档数量（用于测试，None表示处理全部）'
    )
    
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='强制重建索引（即使已存在）'
    )
    
    return parser.parse_args()


def check_requirements():
    """检查必要的依赖"""
    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        logging.info("所有必要依赖已安装")
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            logging.info(f"检测到GPU: {torch.cuda.get_device_name()}")
            logging.info(f"CUDA版本: {torch.version.cuda}")
        else:
            logging.warning("未检测到GPU，将使用CPU进行计算")
            
        # 检查FAISS GPU支持
        try:
            faiss.StandardGpuResources()
            logging.info("FAISS GPU支持可用")
        except:
            logging.warning("FAISS GPU支持不可用，将使用CPU")
            
    except ImportError as e:
        logging.error(f"缺少必要依赖: {e}")
        logging.error("请安装: pip install faiss-gpu torch sentence-transformers")
        sys.exit(1)


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logging.info("开始构建RAG索引")
    
    # 解析参数
    args = parse_arguments()
    logging.info(f"参数配置: {vars(args)}")
    
    # 检查依赖
    check_requirements()
    
    # 检查语料库文件
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        logging.error(f"语料库文件不存在: {corpus_path}")
        sys.exit(1)
    
    logging.info(f"语料库文件: {corpus_path} (大小: {corpus_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # 检查索引目录
    path_config = PathConfig()
    index_dir = Path(path_config.rag_cache_dir)
    if index_dir.exists() and not args.force_rebuild:
        logging.warning(f"索引目录已存在: {index_dir}")
        response = input("是否继续构建？这将覆盖现有索引 (y/N): ")
        if response.lower() != 'y':
            logging.info("用户取消构建")
            sys.exit(0)
    
    try:
        # 创建RAG检索器实例
        logging.info("初始化RAG检索器...")
        retriever = RAGRetriever(
            model_name=args.model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 加载语料库
        logging.info("开始加载FlashRAG语料库...")
        success = retriever.load_flashrag_corpus(
            corpus_path=str(corpus_path),
            max_docs=args.max_docs
        )
        
        if not success:
            logging.error("语料库加载失败")
            sys.exit(1)
        
        load_time = time.time() - start_time
        logging.info(f"语料库加载完成，耗时: {load_time:.2f}秒")
        
        # 构建索引
        logging.info("开始构建FAISS索引...")
        index_start_time = time.time()
        
        success = retriever.build_index()
        
        if not success:
            logging.error("索引构建失败")
            sys.exit(1)
        
        index_time = time.time() - index_start_time
        total_time = time.time() - start_time
        
        # 输出统计信息
        logging.info("="*50)
        logging.info("索引构建完成！")
        logging.info(f"总耗时: {total_time:.2f}秒")
        logging.info(f"语料库加载耗时: {load_time:.2f}秒")
        logging.info(f"索引构建耗时: {index_time:.2f}秒")
        logging.info(f"索引保存路径: {index_dir}")
        logging.info(f"处理文档数量: {len(retriever.doc_metadata) if hasattr(retriever, 'doc_metadata') else 'N/A'}")
        logging.info(f"索引向量数量: {len(retriever.chunk_metadata) if hasattr(retriever, 'chunk_metadata') else 'N/A'}")
        logging.info("="*50)
        
    except KeyboardInterrupt:
        logging.info("用户中断构建过程")
        sys.exit(1)
    except Exception as e:
        logging.error(f"构建过程中发生错误: {e}")
        logging.exception("详细错误信息:")
        sys.exit(1)


if __name__ == '__main__':
    main()