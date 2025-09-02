from flashrag.config import Config
from flashrag.utils import get_retriever
import os

os.environ['HF_HOME'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data'
os.environ['TRANSFORMERS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp" 
os.environ['HF_DATASETS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp"

def single_retrieve_with_rerank(config_path, query):
    # 加载配置（与批量检索共用同一配置文件）
    config = Config(config_path)
    
    # 初始化检索器（支持纯文本、多模态、多路检索）
    retriever = get_retriever(config)
    
    # 执行单条检索（带重排序，参数与批量接口一致）
    # return_score=True 时返回 (检索结果, 分数)，否则仅返回检索结果
    result, score = retriever.search(
        query=query,
        return_score=True  # 是否返回相关性分数
    )
    
    return result, score

if __name__ == "__main__":
    # 单条查询示例
    test_query = "Water boils at 100°C at sea level, is it true?"

    os.environ['HF_HOME'] = '/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data'
    os.environ['TRANSFORMERS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp" 
    os.environ['HF_DATASETS_CACHE'] = "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/wiki_data/tmp"

    
    # 执行单条检索
    retrieval_result, relevance_score = single_retrieve_with_rerank(
        "configs/retrieval_config.yaml",  
        test_query
    )
    
    # 打印结果
    print(retrieval_result)
    print("---")
    print(relevance_score)