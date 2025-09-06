import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

# 处理相对导入和绝对导入的兼容性
try:
    from .conflict_detector import CausalConflictDetector
    from .counterfactual_editor import CounterfactualEditor
    from ..dynamic.fingerprint_builder import DynamicFingerprintBuilder
    from ..dynamic.vector_index import DynamicVectorIndex
    from ..dynamic.rag_retriever import RAGRetriever
    from ..utils.path_config import get_path_config, get_rag_paths
except ImportError:
    # 当直接运行此文件时，使用绝对导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from causal_editor.core.conflict_detector import CausalConflictDetector
    from causal_editor.core.counterfactual_editor import CounterfactualEditor
    from causal_editor.dynamic.fingerprint_builder import DynamicFingerprintBuilder
    from causal_editor.dynamic.vector_index import DynamicVectorIndex
    from causal_editor.dynamic.rag_retriever import RAGRetriever
    from causal_editor.utils.path_config import get_path_config, get_rag_paths

logger = logging.getLogger(__name__)


class CausalEditor:

    def __init__(
        self,
        model_name: str = "llama-2-7b",
        device: str = "cuda",
        model: Optional[torch.nn.Module] = None,
        tokenizer = None,
        rag_config: Optional[Dict[str, Any]] = None,
    ):
       
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.rag_config = rag_config or {}
        # 为检索开关与模式设置安全默认值，避免后续日志/元数据访问时报属性错误
        self.use_rag_retrieval = self.rag_config.get('use_rag_retrieval', True)
        self.retrieval_mode = self.rag_config.get('retrieval_mode', 'rag_only')

        # 获取模型的hidden_size
        self.hidden_size = self._get_hidden_size(self.model, model_name)

        fingerprint_builder_config = self.rag_config.get('fingerprint_builder', {})

        self.fingerprint_builder = DynamicFingerprintBuilder(
            model=self.model,
            tokenizer=self.tokenizer,
            target_layers=fingerprint_builder_config.get('target_layers', []),
            device=device,
            max_batch_size=fingerprint_builder_config.get('max_batch_size', 16),
            fingerprint_dim=fingerprint_builder_config.get('fingerprint_dim', self.hidden_size)
        )

        vector_index_config = self.rag_config.get('vector_index', {})
        
        # 初始化动态向量索引
        self.dynamic_index = DynamicVectorIndex(
            dimension=self.hidden_size,
            device=device,
            index_type=vector_index_config.get('index_type', "hnsw"),
            max_vectors=vector_index_config.get('max_vectors', 5000), #  TODO: 最大向量数量之后可能要改
            similarity_threshold = vector_index_config.get('similarity_threshold', 0.5)
        )
        # 与动态索引的相似度阈值保持一致，供日志与检测使用
        self.similarity_threshold = getattr(self.dynamic_index, 'similarity_threshold', 0.5)

        conflict_detector_config = self.rag_config.get('conflict_detector', {})

        # 初始化冲突检测器和编辑器
        self.conflict_detector = CausalConflictDetector(
            similarity_threshold=conflict_detector_config.get('similarity_threshold', 0.8),
            conflict_threshold=conflict_detector_config.get('conflict_threshold', 0.6),
            enable_dynamic_threshold=conflict_detector_config.get('enable_dynamic_threshold', True),
            threshold_adjustment_factor=conflict_detector_config.get('threshold_adjustment_factor', 0.1)
        )

        counterfactual_editor_config = self.rag_config.get('counterfactual_editor', {})
        
        # 初始化编辑器
        self.counterfactual_editor = CounterfactualEditor(
            edit_strength=counterfactual_editor_config.get('edit_strength', 0.5), 
            device=self.device, 
            min_confidence=counterfactual_editor_config.get('min_confidence', 0.3),
            enable_rag_editing=counterfactual_editor_config.get('enable_rag_editing', True),
            activation_rollback_strength=counterfactual_editor_config.get('activation_rollback_strength', 0.5),
            resampling_temperature=counterfactual_editor_config.get('resampling_temperature', 0.5),
            activation_weighting_factor=counterfactual_editor_config.get('activation_weighting_factor', 0.8),
            hidden_size=self.hidden_size
        )

        # 运行时状态管理
        self.current_layer_id = None
        self.prompt_length = None
        self.is_mc_mode = False

        # Token上下文管理
        self.current_token_context = {
            "generated_tokens": [],
            "context_tokens": [],
            "generation_step": 0,
        }

        # 层级映射（不同模型可能有不同的层级结构）
        self.layer_mapping = self._get_layer_mapping(model_name)
        
        # 预构建的指纹索引缓存
        self.prebuilt_fingerprints = {}
        self.prebuilt_index_ready = False
        self.current_input_doc = []

        # CausalEditor初始化完成
    
    def clear_fingerprints(self):
        """清空当前轮次的激活指纹
        
        在生成结束时调用，清理临时存储的激活指纹
        """
        try:
            # 清空预构建的指纹
            self.prebuilt_fingerprints.clear()
            
            # 清空动态向量索引
            self.dynamic_index.clear_all()
            
            # 清空指纹构建器的临时缓存
            self.fingerprint_builder.clear_rag_cache()
            
            # 重置索引状态
            self.prebuilt_index_ready = False
            self.current_input_doc = []
            
            logging.info("已清空当前轮次的激活指纹")
            
        except Exception as e:
            logging.error(f"清空激活指纹时发生错误: {e}")

    def prepare_for_input(self, user_input_text: str, rag_retriever):
        """
        为新的用户输入做准备，构建指纹索引。
        
        该方法是RAG系统的核心入口点，负责：
        1. 存储当前用户输入
        2. 触发指纹预构建流程
        3. 初始化RAG检索上下文

        Args:
            user_input_text (str): 用户的原始输入文本
        """
        self.current_user_input = user_input_text  # 存储当前输入
        logging.info(f"为输入准备CausalEditor : {user_input_text[:80]}...")
        
        # 预构建指纹索引，支持RAG检索
        success = self.prebuild_fingerprints_for_input(user_input_text, rag_retriever)
        if success:
            logging.info("指纹索引预构建成功")
        else:
            logging.warning("指纹索引预构建失败，将使用实时构建模式")
    
    def get_current_input(self) -> Optional[str]:
        """获取当前存储的用户输入。"""
        return getattr(self, 'current_user_input', None)
    
    def prebuild_fingerprints_for_input(self, input_text: str, rag_retriever) -> bool:
        """
        为输入文本预构建指纹索引
        
        实现RAG系统的核心流程：
        1. 获取RAG检索片段
        2. 对候选内容进行前向传播，构建动态激活指纹
        3. 构建FAISS向量索引库，用于后续冲突检测
        
        Args:
            input_text (str): 用户输入文本
            
        Returns:
            bool: 是否成功构建指纹索引
            
        Raises:
            Exception: 当指纹构建过程中出现错误时
        """
        # 开始为输入文本预构建指纹索引
        try:
            logging.debug(f"开始为输入构建指纹索引: {input_text[:100]}...")
            
            # 1. 使用RAG检索器获取相关文档片段
            candidates = []
            
            try:
                # 使用RAG检索器检索相关文档
                start_time = time.time()
                result, score = rag_retriever.search(
                    query=input_text,
                    return_score=True
                )
                end_time = time.time()
                retrieval_time = end_time - start_time
                print(f"RAG检索耗时: {retrieval_time:.4f}秒")
                
                # 使用双曲正切函数将分数转换为-1到1的范围
                score = [torch.tanh(torch.tensor(s)).item() for s in score]
                
                # 将检索结果转换为候选格式
                for i, doc in enumerate(result):
                    contents = doc.get('contents', '')
                    # 从contents的第一行提取title
                    lines = contents.split('\n')
                    title = lines[0] if lines else ''
                    text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                    
                    candidate = {
                        'text': text,
                        'title': title,
                        'score': score[i] if i < len(score) else 0.0,
                        'source': 'rag_retrieval',
                        'metadata': {'id': doc.get('id', '')},
                        'fragment_id': f"frag_{i}_{doc.get('id', '')}"
                    }
                    candidates.append(candidate)
                
                logging.info(f"RAG检索完成，获得 {len(candidates)} 个候选文档片段")
                
            except Exception as e:
                logging.error(f"RAG检索失败: {e}")
                candidates = []
            
            if not candidates:
                # 未找到候选文档片段，跳过指纹构建
                logging.info(f"未找到相关候选内容，检索模式: {self.retrieval_mode}")
                self.prebuilt_index_ready = False
                return False
            
            # 提取到候选文档片段
            self.current_input_doc = candidates
            
            # 2. 候选内容获取（已在candidate_filter中完成）
            # 3. 对候选内容进行前向传播，构建动态激活指纹
            logging.debug("开始构建动态激活指纹...")
            # 使用用户问题和检索片段构建指纹，不使用缓存
            fingerprints_dict = self.fingerprint_builder.build_rag_fingerprints(
                user_question=input_text,
                retrieved_fragments=candidates,
                use_cache=False
            )
            
            if not fingerprints_dict:
                logging.warning(f"指纹构建失败，候选数量: {len(candidates)}")
                self.prebuilt_index_ready = False
                return False
            
            # 存储预构建的指纹
            self.prebuilt_fingerprints = fingerprints_dict
            logging.debug(f"成功构建 {len(fingerprints_dict)} 个片段的指纹")
            
            # 4. 构建FAISS向量索引库
            logging.debug("开始构建FAISS向量索引...")
            
            # 清空现有索引
            self.dynamic_index.clear_all()
            
            # 重新组织指纹：按层聚合为 [num_fragments, hidden_dim]
            layer_to_vectors = defaultdict(list)
            layer_to_metadata = defaultdict(list)
            
            for i, candidate in enumerate(candidates):
                frag_id = candidate.get("fragment_id")
                if not frag_id:
                    continue
                fragment_fp = fingerprints_dict.get(frag_id)
                if not fragment_fp or not isinstance(fragment_fp, dict):
                    continue
                
                for layer_id, vec in fragment_fp.items():
                    # 兼容不同形状：期望为 [hidden_dim]
                    if isinstance(vec, torch.Tensor):
                        if vec.dim() == 1:
                            processed_vec = vec.detach()
                            layer_to_vectors[layer_id].append(processed_vec)
                        elif vec.dim() == 2 and vec.size(0) == 1:
                            processed_vec = vec.squeeze(0).detach()
                            layer_to_vectors[layer_id].append(processed_vec)
                        else:
                            # 不期望的形状，尽量拉平截断到hidden_size
                            try:
                                flat = vec.reshape(-1)[: self.hidden_size]
                                processed_vec = flat.detach()
                                layer_to_vectors[layer_id].append(processed_vec)
                                logging.debug(f"片段 {i} 层 {layer_id} - 形状调整: {vec.shape} -> {processed_vec.shape}")
                            except Exception as e:
                                logging.warning(f"片段 {i} 层 {layer_id} - 形状调整失败: {e}")
                                continue
                    else:
                        continue
                    
                    # 构建与向量一一对应的元数据
                    metadata = {
                        "entity_id": i,
                        "layer_id": layer_id,
                        "retrieval_mode": self.retrieval_mode,
                        "text_fragment": candidate.get("text", ""),
                        "title": candidate.get("title", ""),
                        "source": candidate.get("source", "rag_retrieval"),
                        "score": candidate.get("score", 0.0),
                    }
                    if candidate.get("metadata"):
                        metadata.update(candidate["metadata"])
                    layer_to_metadata[layer_id].append(metadata)
            
            # 为每一层构建索引
            total_vectors = 0
            for layer_id, vec_list in layer_to_vectors.items():
                if not vec_list:
                    continue
                # [num_fragments, hidden_dim]
                layer_fingerprints = torch.stack(vec_list, dim=0).to(self.device)
                     
                # 添加向量到索引
                self.dynamic_index.add_vectors(
                    layer_id=layer_id,
                    vectors=layer_fingerprints,
                    metadata_list=layer_to_metadata.get(layer_id, [{}] * layer_fingerprints.size(0)),
                )
                total_vectors += layer_fingerprints.size(0)
            
            self.prebuilt_index_ready = True
            logging.info(f"指纹索引构建完成，总向量数: {total_vectors}，检索模式: {self.retrieval_mode}")
            return True
            
        except Exception as e:
            logging.error(f"预构建指纹索引失败: {e}")
            logging.error(f"错误详情: {str(e)}")
            # 重置相关状态
            self.prebuilt_index_ready = False
            self.current_input_doc = []
            self.prebuilt_fingerprints = {}
            return False

    def _get_hidden_size(self, model: Optional[torch.nn.Module], model_name: str) -> int:
        """
        动态获取模型的隐藏维度
        
        Args:
            model: 模型实例
            model_name: 模型名称
            
        Returns:
            隐藏维度大小
        """
        if model is not None:
            # 尝试从模型配置中获取
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                return model.config.hidden_size
            elif hasattr(model, 'config') and hasattr(model.config, 'n_embd'):
                return model.config.n_embd
            elif hasattr(model, 'config') and hasattr(model.config, 'd_model'):
                return model.config.d_model
        
        # 根据模型名称推断
        model_name_lower = model_name.lower()
        if 'tinyllama' in model_name_lower:
            return 2048
        elif 'llama-7b' in model_name_lower or 'llama2-7b' in model_name_lower:
            return 4096
        elif 'llama-13b' in model_name_lower or 'llama2-13b' in model_name_lower:
            return 5120
        elif 'mistral' in model_name_lower:
            return 4096
        elif 'gpt2' in model_name_lower:
            return 768
        else:
            # 默认值，但会记录警告
            logging.warning(f"无法确定模型 {model_name} 的隐藏维度，使用默认值 4096")
            return 4096
    
    def _get_layer_mapping(self, model_name: str) -> Dict[str, int]:
        """
        获取模型层级映射
        不同模型的层级结构可能不同，需要适配
        """
        if "llama" in model_name.lower():
            return {"attn_factor": 2, "mlp_offset": 1}
        elif "mistral" in model_name.lower():
            return {"attn_factor": 2, "mlp_offset": 1}
        else:
            # 默认映射
            return {"attn_factor": 2, "mlp_offset": 1}

    def set_generation_mode(
        self, is_mc: bool = False, prompt_length: Optional[int] = None
    ):
        """
        设置生成模式

        Args:
            is_mc: 是否为multiple choice模式
            prompt_length: 提示长度（MC模式需要）
        """
        self.is_mc_mode = is_mc
        self.prompt_length = prompt_length
        # 设置生成模式

    def set_current_layer(self, layer_id: str):
        """
        设置当前处理的层

        Args:
            layer_id: 层ID，格式如 "10.attn" 或 "10.mlp"
        """
        self.current_layer_id = layer_id

    def should_edit_layer(self, layer_id: str) -> bool:
        """
        判断是否应该编辑当前层

        Args:
            layer_id: 层ID，可能的格式："6", "6.attn", "model.layers.6", "model.layers.6.self_attn"

        Returns:
            是否应该编辑
        """
        try:
            # 处理不同格式的层ID
            if "model.layers." in layer_id:
                # 格式："model.layers.6" 或 "model.layers.6.self_attn"
                parts = layer_id.split(".")
                layer_num = int(parts[2])  # 提取layers后面的数字
            elif "." in layer_id:
                # 格式："6.attn" 或 "6.mlp"
                layer_num = int(layer_id.split(".")[0])
            else:
                # 格式："6"
                layer_num = int(layer_id)
            
            # 检查是否在中间层范围内
            return layer_num in self.fingerprint_builder.target_layers
        except (ValueError, IndexError) as e:
            logging.warning(f"无法解析层ID: {layer_id}, 错误: {e}")
            return False

    def update_token_context(
        self,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
    ):
        """
        更新token上下文信息 - 新增方法

        Args:
            generated_tokens: 新生成的tokens
            context_tokens: 上下文tokens
        """
        if generated_tokens is not None:
            self.current_token_context["generated_tokens"] = (
                generated_tokens[-5:] if len(generated_tokens) > 5 else generated_tokens
            )
            self.current_token_context["generation_step"] += 1

        if context_tokens is not None:
            self.current_token_context["context_tokens"] = (
                context_tokens[-10:] if len(context_tokens) > 10 else context_tokens
            )

    @torch.inference_mode()
    def edit_activations(
        self,
        activations: torch.Tensor,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
        input_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        编辑激活状态的主入口函数（优化版本）
        
        现在使用预构建的指纹索引，不会导致递归调用

        Args:
            activations: 当前激活状态 [batch_size, seq_len, hidden_dim]
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            input_text: 输入文本（动态模式需要）

        Returns:
            编辑后的激活状态
        """
        try:
            
            # 更新token上下文
            self.update_token_context(generated_tokens, context_tokens)

            if self.current_layer_id is None:
                logging.warning("当前层ID未设置，跳过编辑")
                return activations

            # 检查是否应该编辑当前层
            should_edit = self.should_edit_layer(self.current_layer_id)
            # 检查是否应该编辑当前层
            if not should_edit:
                return activations

            # 使用更新后的token上下文进行冲突检测和编辑
            current_generated = (
                generated_tokens or self.current_token_context["generated_tokens"]
            )
            current_context = context_tokens or self.current_token_context["context_tokens"]

            # 调用detect_and_edit方法进行动态检测和编辑
            edited_activations, conflict_info = self.detect_and_edit(
                activations=activations,
                layer_id=self.current_layer_id,
                generated_tokens=current_generated,
                context_tokens=current_context,
                input_text=input_text,
            )

            # 冲突检测和编辑完成

            return edited_activations
            
        except Exception as e:
            logging.error(f"编辑激活状态时发生错误: {e}")
            return activations
    
    def finish_generation(self):
        """完成生成，清理临时资源
        
        在每轮问答生成完成后调用，清空激活指纹缓存
        """
        self.clear_fingerprints()
        logging.info("生成完成，已清理临时激活指纹")

    def _dynamic_detect_conflict(
        self,
        activations: torch.Tensor,
        layer_id: str,
        input_text: Optional[str] = None,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        动态模式的冲突检测（优化版本）
        
        现在只进行向量库查询，不再调用模型前向传播
        
        Args:
            activations: 当前激活状态
            layer_id: 层ID
            input_text: 输入文本
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            
        Returns:
            冲突信息字典
        """
        # 检查是否有预构建的指纹索引
        if not self.prebuilt_index_ready:
            logging.debug("预构建索引未就绪，跳过冲突检测")
            return {"has_conflict": False, "confidence": 0.0}
            
        try:
            # 安全处理激活状态的维度
            try:
                logger.debug(f"🔍 原始激活形状: {activations.shape}")
                
                if activations.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                    # 取最后一个token的激活
                    current_activation = activations[:, -1, :].squeeze()  # [hidden_dim]
                    # current_activation = activations.mean(dim=1).squeeze()  # [hidden_dim]
                elif activations.dim() == 2:  # [seq_len, hidden_dim]
                    # 取最后一个token的激活
                    current_activation = activations[-1, :].squeeze()  # [hidden_dim]
                    # current_activation = activations.mean(dim=0)  # [hidden_dim]
                else:  # [hidden_dim]
                    current_activation = activations
                
                # 确保是1D张量
                if current_activation.dim() > 1:
                    current_activation = current_activation.flatten()
                
                # 检查张量是否有效
                if current_activation.numel() == 0:
                    return {"has_conflict": False, "confidence": 0.0}
                    
            except Exception as e:
                logging.warning(f"激活状态处理失败: {e}")
                return {"has_conflict": False, "confidence": 0.0}
            
            # 使用预构建的FAISS索引进行查询
            logger.debug(f"🔍 正在搜索层 {layer_id} 的相似向量，激活维度: {current_activation.shape}")
            logger.debug(f"🔍 当前激活向量范围: [{current_activation.min():.4f}, {current_activation.max():.4f}]")
            search_results = self.dynamic_index.search(
                layer_id=layer_id,
                query_vector=current_activation,  # 直接使用1D向量
                k=3  # 获取最相似的3个结果
            )
            
            logger.debug(f"🔍 层 {layer_id} 搜索结果: {len(search_results)} 个结果")
            if search_results:
                for i, result in enumerate(search_results[:3]):  # 只显示前3个结果
                    similarity = result.get("similarity_score", 0.0)
                    logger.debug(f"  📊 结果 {i+1}: 相似度={similarity:.4f}, 阈值={self.similarity_threshold}")
            
            if not search_results or len(search_results) == 0:
                logger.debug(f"❌ 层 {layer_id} 没有找到匹配结果")
                return {"has_conflict": False, "confidence": 0.0}
            
            # 获取最佳匹配
            best_result = search_results[0]
            max_similarity = best_result.get("similarity_score", 0.0)
            
            logger.debug(f"✨ 最佳匹配: 相似度={max_similarity:.4f}")
            
            # 获取修正向量 - 从向量库中获取对应的激活
            correction_vector = current_activation  
            
            # 尝试从预构建的指纹中获取对应的向量
            if hasattr(self, 'prebuilt_fingerprints') and self.prebuilt_fingerprints:
                # 从metadata中获取fragment相关信息
                entity_id = best_result.get("entity_id")
                if entity_id is not None and hasattr(self, 'current_input_doc') and self.current_input_doc:
                    # 通过entity_id找到对应的candidate
                    if entity_id < len(self.current_input_doc):
                        candidate = self.current_input_doc[entity_id]
                        frag_id = candidate.get("fragment_id")
                        if frag_id and frag_id in self.prebuilt_fingerprints:
                            fragment_fp = self.prebuilt_fingerprints[frag_id]
                            if layer_id in fragment_fp:
                                correction_vector = fragment_fp[layer_id]
                                logging.debug(f"使用向量库中的激活，frag_id: {frag_id}")
                            else:
                                logging.debug(f"层 {layer_id} 在片段 {frag_id} 的指纹中不存在")
                        else:
                            logging.debug(f"片段 {frag_id} 在预构建指纹中不存在")
                    else:
                        logging.debug(f"entity_id {entity_id} 超出候选范围")
                else:
                    logging.debug("无法获取entity_id或current_input_doc")
            
            
            # 有有效修正向量时，放宽相似度要求
            if max_similarity < self.conflict_detector.conflict_threshold:
                logger.debug(f"❌ 相似度 {max_similarity:.4f} 过低，即使有修正向量也跳过")
                return {"has_conflict": False, "confidence": 0.0}
            
            # 构建片段指纹和分数字典，用于RAG冲突检测
            fragment_fingerprints = getattr(self, 'prebuilt_fingerprints', {})
            fragment_scores = {}
            
            # 从fingerprint_builder的缓存中获取分数信息
            if fragment_fingerprints and hasattr(self, 'fingerprint_builder'):
                for frag_id in fragment_fingerprints.keys():
                    fragment_scores[frag_id] = self.fingerprint_builder.get_fragment_score(frag_id)
            
            # 使用更优化的RAG冲突检测方法
            conflict_result = self.conflict_detector.detect_rag_conflict(
                current_activation=current_activation,
                fragment_fingerprints=fragment_fingerprints,
                fragment_scores=fragment_scores,
                layer_id=layer_id,
                generated_tokens=generated_tokens,
                context_tokens=context_tokens
            )
            
            # 添加额外的检索信息
            conflict_result.update({
                "correction_vector": correction_vector,
                "fragment_fingerprints": fragment_fingerprints,
                "fragment_scores": fragment_scores,
                "retrieved_knowledge": [{
                    "similarity_score": result.get("similarity_score", 0.0),
                    "confidence": result.get("similarity_score", 0.0)  # 使用相似度作为置信度
                } for result in search_results[:3]]  # 返回前3个检索结果
            })
            
            return conflict_result
            
        except Exception as e:
            logging.warning(f"动态冲突检测失败: {e}")
            return {"has_conflict": False, "confidence": 0.0}

    def detect_and_edit(
        self,
        activations: torch.Tensor,
        layer_id: str,
        generated_tokens: Optional[List[str]] = None,
        context_tokens: Optional[List[str]] = None,
        input_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        检测冲突并执行编辑

        Args:
            activations: 当前激活状态 [batch_size, seq_len, hidden_dim]
            layer_id: 当前层ID
            generated_tokens: 已生成的tokens
            context_tokens: 上下文tokens
            input_text: 输入文本（动态模式需要）

        Returns:
            编辑后的激活状态和冲突信息
        """
        # 检测冲突并执行编辑
        
        # 设置当前层
        self.set_current_layer(layer_id)

        # 动态模式：实时生成和检测
        conflict_info = self._dynamic_detect_conflict(
            activations=activations,
            layer_id=layer_id,
            input_text=input_text,
            generated_tokens=generated_tokens,
            context_tokens=context_tokens
        )

        # logging.debug(f"冲突检测结果: has_conflict={conflict_info.get('has_conflict', False)}, confidence={conflict_info.get('confidence', 0.0):.3f}")
        
        # 如果检测到冲突且当前层应该被编辑
        if conflict_info["has_conflict"] and self.should_edit_layer(layer_id):
            # 执行RAG反事实编辑
            edited_activations = self.counterfactual_editor.edit_rag(
                activations=activations,
                conflict_info=conflict_info,
                fragment_fingerprints=conflict_info.get('fragment_fingerprints', {}),
                fragment_scores=conflict_info.get('fragment_scores', {}),
                layer_id=layer_id,
            )

            # 在层执行了编辑

            return edited_activations, conflict_info
        else:
            # logging.debug(f"层 {layer_id} 未执行编辑")
            return activations, conflict_info

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取运行统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "model_name": self.model_name,
            "conflict_detector_stats": self.conflict_detector.get_statistics(),
            "counterfactual_editor_stats": self.counterfactual_editor.get_statistics(),
        }
        
        if self.dynamic_index:
            dynamic_stats = self.dynamic_index.get_statistics()
            stats["dynamic_index_size"] = dynamic_stats.get('total_vectors', 0)
            stats["dynamic_index_stats"] = dynamic_stats
        else:
            stats["dynamic_index_size"] = 0
        
            
        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.conflict_detector.reset_statistics()
        self.counterfactual_editor.reset_statistics()

    def save_config(self, path: str):
        """
        保存配置

        Args:
            path: 保存路径
        """
        config = {
            "model_name": self.model_name,
            "edit_strength": self.edit_strength,
            "num_middle_layers": self.num_middle_layers,
            "similarity_threshold": self.conflict_detector.similarity_threshold,
            "conflict_threshold": self.conflict_detector.conflict_threshold,
            "layer_mapping": self.layer_mapping,
        }

        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 配置已保存

    def get_rag_config(self) -> Dict[str, Any]:
        """
        获取RAG配置信息
        
        Returns:
            RAG配置字典
        """
        return {
            "rag_config": self.rag_config.copy() if self.rag_config else {}
        }
    