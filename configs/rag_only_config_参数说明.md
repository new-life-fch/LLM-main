{
  "model": {
    "name": "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/llama2-7b-chat-hf",
    "device": "cuda",
    "batch_size": 8,
    "dtype": "float16"
  },
  "knowledge_extraction": {
    "type": "rag",
    "rate_limit": 1.0,
    "cache_dir": "./cache/rag"
  },
  "rag_retrieval": {
    "config": "configs/retrieval_config.yaml"
  },
  "reranker_config": {
    "enabled": true,
    "model_name": "/root/autodl-tmp/LLM-main/LLM-main/New_Project-CausalEdit/model/bge-reranker-base",
    "final_top_k": 3,
    "batch_size": 512
  },
  "fallback_config": {
    "enable_fallback": true,
    "fallback_threshold_high": 0.6,
    "fallback_threshold_medium": 0.4,
    "fallback_threshold_low": 0.2,
    "enable_dynamic_threshold": true,
    "fallback_cache_ttl": 864000,
    "min_threshold": 0.1,
    "max_threshold": 0.6,
    "threshold_adjustment_interval": 50,
    "fallback_cache_path": "./cache/rag/fallback_cache.db"
  },
  "wikipedia_data": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "language": "en",
    "date": "20231101",
    "preprocessing": {
      "remove_tables": true,
      "remove_infoboxes": true,
      "min_text_length": 100,
      "max_text_length": 2048
    }
  },
  "performance_monitoring": {
    "enabled": true,
    "log_retrieval_stats": true,
    "log_performance_metrics": true,
    "stats_update_interval": 100
  },
  "fingerprint_builder": {
    "target_layers": [
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20
    ],
    "cache_size": 500,
    "max_batch_size": 8,
    "fingerprint_dim": 4096
  },
  "vector_index": {
    "index_type": "flat",
    "similarity_metric": "cosine",
    "hnsw_m": 64,
    "hnsw_ef_construction": 400,
    "hnsw_ef_search": 200,
    "max_vectors": 500000,
    "similarity_threshold": 0.5 **生成时问题向量查询指纹库时的查询阈值**
  },
  "conflict_detector": {
    "similarity_threshold": 0.75, **编辑激活时的相似度阈值**
    "conflict_threshold": 0.5,
    "enable_dynamic_threshold": true,
    "threshold_adjustment_factor": 0.5, **阈值调节因子，动态调节阈值时使用，控制阈值调整幅度**
    "token_window": 64
  },
  "counterfactual_editor": {
    "edit_strength": 3,
    "min_confidence": 0.5, **编辑激活时的最小置信度阈值，不超过阈值，不能编辑**
    "enable_rag_editing": true,
    "activation_rollback_strength": 0.8, **激活回退：将当前激活向量回退到检索片段的激活状态，控制回退强度**
    "resampling_temperature": 1.2,
    "activation_weighting_factor": 0.6
  },
  "evaluation": {
    "truthfulqa_data": "TruthfulQA/data/TruthfulQA.csv",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "do_sample": true
  },
  "generation": {
    "max_new_tokens": 4096,
    "temperature": 0.6,
    "do_sample": true,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "num_beams": 1,
    "early_stopping": true,
    "is_mc_mode": false,
    "prompt_length": 1000
  }
}