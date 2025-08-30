# RAG Only Config 参数详细说明

本文档详细解释 `rag_only_config.json` 配置文件中每个参数的作用和控制功能。

## 📋 配置文件结构概览

```
rag_only_config.json
├── model                    # 模型配置
├── knowledge_extraction     # 知识抽取配置
├── rag_retrieval           # RAG检索配置
├── reranker_config         # 重排序器配置
├── fallback_config         # 回退机制配置
├── wikipedia_data          # Wikipedia数据配置
├── web_knowledge_retrieval # 网络知识检索配置
├── performance_monitoring  # 性能监控配置
├── fingerprint_building    # 指纹构建配置
├── vector_database         # 向量数据库配置
├── causal_editor          # 因果编辑器配置
└── evaluation             # 评估配置
```

## 🤖 model - 模型配置

控制基础语言模型的加载和运行参数。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `name` | string | 模型路径或HuggingFace模型名称 | `/path/to/model` 或 `meta-llama/Llama-2-7b-chat-hf` |
| `device` | string | 运行设备：`cuda`(GPU) 或 `cpu` | `"cuda"` |
| `batch_size` | int | 批处理大小，影响内存使用和推理速度 | `8` |
| `torch_dtype` | string | 数据类型：`float16`节省内存，`float32`精度更高 | `"float16"` |

## 🔍 knowledge_extraction - 知识抽取配置

控制知识获取的方式和限制。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `type` | string | 知识抽取类型：`rag`(检索增强生成) 或 `kg`(知识图谱) | `"rag"` |
| `rate_limit` | float | 请求频率限制(秒)，防止过度调用外部API | `1.0` |
| `cache_dir` | string | 缓存目录，存储检索结果以提高效率 | `"./cache/rag"` |

## 📚 rag_retrieval - RAG检索配置

控制文档检索的核心参数。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `enabled` | bool | 是否启用RAG检索功能 | `true` |
| `model_name` | string | 文本嵌入模型，用于计算文本相似度 | `"BAAI/bge-large-en-v1.5"` |
| `top_k` | int | 检索返回的文档数量，越大覆盖面越广但噪音可能增加 | `3` |
| `min_score` | float | 最小相似度阈值，低于此值的文档将被过滤 | `0.5` |
| `index_path` | string | FAISS向量索引文件路径，用于快速检索 | `"./cache/rag/faiss_index"` |
| `documents_path` | string | SQLite文档数据库路径，存储原始文档内容 | `"./cache/rag/documents.db"` |
| `cache_dir` | string | RAG系统缓存目录 | `"./cache/rag"` |
| `wikipedia_dataset` | string | Wikipedia数据集名称 | `"wikimedia/wikipedia"` |
| `max_seq_length` | int | 最大文本序列长度，影响编码效果 | `512` |
| `batch_size` | int | 编码批处理大小，影响内存使用和速度 | `512` |
| `embedding_dim` | int | 嵌入向量维度，BGE-large-en为1024维 | `1024` |
| `hnsw_m` | int | HNSW索引连接数，影响索引质量和大小 | `96` |
| `hnsw_ef_construction` | int | HNSW构建时搜索范围，影响构建质量 | `800` |
| `hnsw_ef_search` | int | HNSW搜索时候选数量，影响搜索精度 | `400` |
| `enable_sharding` | bool | 是否启用索引分片，用于超大规模数据 | `false` |
| `shard_size` | int | 每个分片的大小（文档数量） | `2000000` |

## 🔄 reranker_config - 重排序器配置

对检索结果进行二次排序以提高质量。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `enabled` | bool | 是否启用重排序功能 | `true` |
| `model_name` | string | 重排序模型，专门用于文档相关性排序 | `"BAAI/bge-reranker-large"` |
| `initial_candidates` | int | 初始候选文档数量，从中选择最相关的 | `500` |
| `final_top_k` | int | 重排序后最终返回的文档数量 | `3` |
| `batch_size` | int | 重排序批处理大小，影响重排序速度 | `256` |
| `use_fp16` | bool | 是否使用半精度浮点数，提升速度节省内存 | `true` |

## 🔙 fallback_config - 回退机制配置

当检索质量不佳时的处理策略。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `enabled` | bool | 是否启用回退机制 | `true` |
| `threshold_high` | float | 高质量阈值，超过此值认为检索结果可靠 | `0.35` |
| `threshold_medium` | float | 中等质量阈值，介于高低阈值之间启用混合检索 | `0.2` |
| `threshold_low` | float | 低质量阈值，低于此值一定启用fallback | `0.2` |
| `enable_dynamic_threshold` | bool | 是否启用动态阈值调整，根据历史表现自动优化 | `true` |
| `cache_ttl` | int | Fallback缓存生存时间（秒），24小时=86400 | `86400` |
| `min_threshold` | float | 动态调整的最小阈值限制 | `0.15` |
| `max_threshold` | float | 动态调整的最大阈值限制 | `0.5` |
| `adjustment_interval` | int | 阈值调整间隔（查询次数） | `100` |

## 📖 wikipedia_data - Wikipedia数据配置

控制Wikipedia数据的处理和分块策略。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `chunk_size` | int | 文档分块大小（字符数），影响检索粒度 | `512` |
| `chunk_overlap` | int | 分块重叠大小，保持上下文连续性 | `50` |
| `language` | string | Wikipedia语言版本 | `"en"` |
| `date` | string | Wikipedia数据快照日期 | `"20231101"` |
| `preprocessing.remove_tables` | bool | 是否移除表格内容 | `true` |
| `preprocessing.remove_infoboxes` | bool | 是否移除信息框内容 | `true` |
| `preprocessing.min_text_length` | int | 最小文本长度，过滤过短内容 | `100` |
| `preprocessing.max_text_length` | int | 最大文本长度，截断过长内容 | `2048` |

## 🌐 web_knowledge_retrieval - 网络知识检索配置

控制在线知识源的使用。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `enabled` | bool | 是否启用网络知识检索作为补充 | `true` |
| `strategy` | string | 检索策略：HYBRID(混合)、WIKIPEDIA_ONLY等 | `"HYBRID"` |
| `max_results` | int | 最大网络检索结果数量 | `10` |
| `timeout` | int | 网络请求超时时间（秒） | `30` |
| `cache_enabled` | bool | 是否启用网络检索结果缓存 | `true` |
| `rate_limit` | float | 请求频率限制（秒/请求） | `2.0` |
| `retry_attempts` | int | 失败重试次数 | `3` |

## 📊 performance_monitoring - 性能监控配置

控制系统性能监控和统计收集。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `enabled` | bool | 是否启用性能监控 | `true` |
| `log_retrieval_stats` | bool | 是否记录检索统计信息 | `true` |
| `log_performance_metrics` | bool | 是否记录性能指标 | `true` |
| `stats_update_interval` | int | 统计信息更新间隔（查询次数） | `100` |

## 🔖 fingerprint_building - 指纹构建配置

控制模型内部表示的提取和分析。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `target_layers` | int | 目标层数，指定从模型的哪些层提取特征 | `10` |
| `cache_dir` | string | 指纹缓存目录 | `"./cache/fingerprints"` |
| `batch_size` | int | 指纹构建时的批处理大小 | `8` |

## 🗄️ vector_database - 向量数据库配置

控制向量索引的构建和搜索策略。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `index_type` | string | 索引类型：`HNSW`(分层导航小世界)算法，平衡速度和精度 | `"HNSW"` |
| `similarity_metric` | string | 相似度计算方法：`cosine`(余弦相似度)或`euclidean`(欧几里得距离) | `"cosine"` |
| `hnsw_m` | int | HNSW参数：每个节点的最大连接数，影响索引质量和大小 | `32` |
| `hnsw_ef_construction` | int | 构建时的搜索范围，越大构建越慢但质量越高 | `200` |
| `hnsw_ef_search` | int | 搜索时的候选数量，影响搜索精度和速度 | `100` |

## ✏️ causal_editor - 因果编辑器配置

控制模型知识编辑的核心参数。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `edit_strength` | float | 编辑强度，控制知识修改的程度 | `1.5` |
| `top_layers` | int | 编辑的目标层数，通常选择模型的高层 | `10` |
| `similarity_threshold` | float | 相似度阈值，用于判断是否需要编辑 | `0.8` |
| `conflict_threshold` | float | 冲突检测阈值，识别知识冲突的敏感度 | `0.65` |
| `min_confidence` | float | 最小置信度，低于此值的编辑将被拒绝 | `0.3` |
| `use_rag_retrieval` | bool | 是否在编辑过程中使用RAG检索 | `true` |
| `retrieval_mode` | string | 检索模式：`rag_only`(仅RAG)、`kg_only`(仅知识图谱)、`hybrid`(混合) | `"rag_only"` |

## 📊 evaluation - 评估配置

控制模型性能评估的参数。

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `truthfulqa_data` | string | 真实性评估数据集路径 | `"TruthfulQA/data/TruthfulQA.csv"` |
| `max_new_tokens` | int | 生成文本的最大长度 | `100` |
| `temperature` | float | 生成温度，控制输出的随机性(0-1，越高越随机) | `0.7` |
| `do_sample` | bool | 是否使用采样生成，`false`则使用贪婪解码 | `true` |

## 🔧 参数调优建议

### 性能优化
- **内存不足**：降低 `batch_size`，使用 `float16`，启用 `enable_sharding`
- **速度优化**：增加 `batch_size`，减少 `top_k`，调整 `hnsw_ef_search`
- **精度提升**：增加 `top_k`，降低 `min_score`，提高 `hnsw_ef_construction`

### RAG检索优化
- **检索质量**：调整 `min_score` 和 `top_k` 的平衡，优化 `hnsw_m` 参数
- **重排序效果**：启用 `reranker_config`，调整 `initial_candidates` 和 `batch_size`
- **动态适应**：启用 `enable_dynamic_threshold`，调整 `adjustment_interval`
- **大规模数据**：启用 `enable_sharding`，调整 `shard_size`

### Wikipedia数据优化
- **分块策略**：根据任务调整 `chunk_size` 和 `chunk_overlap`
- **数据质量**：设置合适的 `min_text_length` 和 `max_text_length`
- **预处理**：根据需求启用/禁用表格和信息框移除

### Fallback机制优化
- **阈值设置**：根据数据质量调整三级阈值
- **缓存策略**：设置合适的 `cache_ttl`
- **动态调整**：启用动态阈值并设置合理的调整间隔

### 因果编辑优化
- **编辑强度**：根据任务需求调整 `edit_strength`
- **冲突检测**：调整 `conflict_threshold` 和 `similarity_threshold`
- **置信度控制**：设置合适的 `min_confidence`

## ⚠️ 注意事项

1. **路径配置**：确保所有路径存在且有读写权限
2. **模型兼容性**：确认嵌入模型和重排序模型的兼容性
3. **资源需求**：根据硬件配置调整批处理大小和模型精度
4. **缓存管理**：定期清理缓存目录以释放存储空间
5. **参数平衡**：检索质量和速度之间需要找到平衡点