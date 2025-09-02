"""统一路径配置管理器

本模块提供项目中所有路径的统一管理，解决相对路径和绝对路径混用的问题。
确保所有组件使用一致的路径配置。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging



class PathConfig:
    """统一路径配置管理器
    
    负责管理项目中所有路径配置，提供统一的路径获取接口，
    确保路径的一致性和正确性。
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """初始化路径配置管理器
        
        Args:
            project_root: 项目根目录，如果为None则自动检测
        """
        # 确定项目根目录
        if project_root:
            self.project_root = Path(project_root).resolve()
        else:
            # 自动检测项目根目录（查找包含causal_editor目录的父目录）
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent.parent
            
        # 确保项目根目录存在
        if not self.project_root.exists():
            raise ValueError(f"项目根目录不存在: {self.project_root}")
            
        # 定义所有路径配置
        self._init_paths()
        
        logging.info(f"路径配置管理器初始化完成，项目根目录: {self.project_root}")
    
    def _init_paths(self):
        """初始化所有路径配置"""
        # 基础目录
        self.cache_root = self.project_root / "cache"
        self.data_root = self.project_root / "data"
        self.config_root = self.project_root / "configs"
        self.log_root = self.project_root / "logs"
        
        # RAG相关路径
        self.rag_cache_dir = self.project_root / "wiki_data" / "indexes" / "wiki_bge_large_ivfpq"
        self.rag_index_path = self.rag_cache_dir / "faiss_index"
        self.rag_documents_path = self.rag_cache_dir / "documents.db"
        self.rag_index_bin = self.rag_cache_dir / "faiss_index.bin"
        self.rag_index_ids = self.rag_cache_dir / "faiss_index.ids"
        
        # Web知识缓存路径
        self.web_knowledge_cache_dir = self.cache_root / "web_knowledge"
        self.web_knowledge_db = self.web_knowledge_cache_dir / "knowledge_cache.db"
        
        # 数据目录路径
        self.wikipedia_data_dir = self.data_root / "wikipedia"
        self.knowledge_base_path = self.data_root / "knowledge_base.json"
        
        # 配置文件路径
        self.main_config_path = self.config_root / "causal_editor_config.json"
        self.rag_config_path = self.config_root / "rag_config.json"
        
        # 临时文件路径
        self.temp_dir = self.cache_root / "temp"
        
        # 日志路径
        self.log_dir = self.project_root / "logs"
        self.main_log_path = self.log_dir / "causal_editor.log"
        
        # 结果输出路径
        self.result_root = self.project_root / "result"
        self.test_results_dir = self.result_root / "test_results"
        self.evaluation_results_dir = self.result_root / "evaluation"
        self.debug_results_dir = self.result_root / "debug"
        
    def get_path(self, path_name: str) -> Path:
        """获取指定路径
        
        Args:
            path_name: 路径名称
            
        Returns:
            Path对象
            
        Raises:
            ValueError: 如果路径名称不存在
        """
        if hasattr(self, path_name):
            return getattr(self, path_name)
        else:
            raise ValueError(f"未知的路径名称: {path_name}")
    
    def get_path_str(self, path_name: str) -> str:
        """获取指定路径的字符串表示
        
        Args:
            path_name: 路径名称
            
        Returns:
            路径字符串
        """
        return str(self.get_path(path_name))
    
    def create_directories(self, *path_names: str):
        """创建指定的目录
        
        Args:
            *path_names: 要创建的目录路径名称
        """
        for path_name in path_names:
            path = self.get_path(path_name)
            if path_name.endswith('_path') or path_name.endswith('_file'):
                # 如果是文件路径，创建其父目录
                path.parent.mkdir(parents=True, exist_ok=True)
                logging.debug(f"创建目录: {path.parent}")
            else:
                # 如果是目录路径，直接创建
                path.mkdir(parents=True, exist_ok=True)
                logging.debug(f"创建目录: {path}")
    
    def get_rag_config(self) -> Dict[str, str]:
        """获取RAG相关的路径配置
        
        Returns:
            RAG路径配置字典
        """
        return {
            "cache_dir": str(self.rag_cache_dir),
            "index_path": str(self.rag_index_path),
            "documents_path": str(self.rag_documents_path),
            "index_bin_path": str(self.rag_index_bin),
            "index_ids_path": str(self.rag_index_ids)
        }
    
    def get_web_knowledge_config(self) -> Dict[str, str]:
        """获取Web知识相关的路径配置
        
        Returns:
            Web知识路径配置字典
        """
        return {
            "cache_dir": str(self.web_knowledge_cache_dir),
            "knowledge_db_path": str(self.web_knowledge_db)
        }
    
    def get_data_config(self) -> Dict[str, str]:
        """获取数据相关的路径配置
        
        Returns:
            数据路径配置字典
        """
        return {
            "data_root": str(self.data_root),
            "wikipedia_data_dir": str(self.wikipedia_data_dir),
            "knowledge_base_path": str(self.knowledge_base_path)
        }
    
    def get_all_paths(self) -> Dict[str, str]:
        """获取所有路径配置
        
        Returns:
            所有路径配置字典
        """
        paths = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, Path):
                    paths[attr_name] = str(attr_value)
        return paths
    
    def validate_paths(self) -> Dict[str, bool]:
        """验证所有路径的有效性
        
        Returns:
            路径验证结果字典
        """
        results = {}
        all_paths = self.get_all_paths()
        
        for path_name, path_str in all_paths.items():
            path = Path(path_str)
            if path_name.endswith('_dir') or path_name.endswith('_root'):
                # 目录路径：检查是否存在或可以创建
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    results[path_name] = True
                except Exception as e:
                    logging.error(f"无法创建目录 {path}: {e}")
                    results[path_name] = False
            elif path_name.endswith('_path') or path_name.endswith('_file'):
                # 文件路径：检查父目录是否存在或可以创建
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    results[path_name] = True
                except Exception as e:
                    logging.error(f"无法创建文件父目录 {path.parent}: {e}")
                    results[path_name] = False
            else:
                # 其他路径：检查是否存在
                results[path_name] = path.exists()
        
        return results
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PathConfig(project_root={self.project_root})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"PathConfig(project_root={self.project_root}, paths={len(self.get_all_paths())})"


# 全局路径配置实例
_global_path_config = None


def get_path_config(project_root: Optional[str] = None) -> PathConfig:
    """获取全局路径配置实例
    
    Args:
        project_root: 项目根目录（仅在首次调用时有效）
        
    Returns:
        PathConfig实例
    """
    global _global_path_config
    if _global_path_config is None:
        _global_path_config = PathConfig(project_root)
    return _global_path_config


def reset_path_config():
    """重置全局路径配置实例"""
    global _global_path_config
    _global_path_config = None


# 便捷函数
def get_rag_paths() -> Dict[str, str]:
    """获取RAG相关路径"""
    return get_path_config().get_rag_config()


def get_web_knowledge_paths() -> Dict[str, str]:
    """获取Web知识相关路径"""
    return get_path_config().get_web_knowledge_config()


def get_data_paths() -> Dict[str, str]:
    """获取数据相关路径"""
    return get_path_config().get_data_config()


def get_log_paths() -> Dict[str, str]:
    """获取日志相关路径配置
    
    Returns:
        日志路径配置字典
    """
    config = get_path_config()
    return {
        "log_dir": str(config.log_root),
        "app_log_path": str(config.log_root / "app.log"),
        "error_log_path": str(config.log_root / "error.log"),
        "debug_log_path": str(config.log_root / "debug.log"),
        "access_log_path": str(config.log_root / "access.log")
    }


def create_all_directories():
    """创建所有必要的目录"""
    path_config = get_path_config()
    path_config.create_directories(
        'cache_root', 'data_root', 'config_root',
        'rag_cache_dir', 'web_knowledge_cache_dir',
        'wikipedia_data_dir', 'temp_dir', 'log_dir'
    )