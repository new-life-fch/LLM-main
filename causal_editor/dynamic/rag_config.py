"""RAG配置管理器

本模块提供RAG系统的配置文件导入功能。
"""

import json
import logging
import os
from typing import Dict, Any, Optional



class RAGConfig:
    """RAG配置管理器
    
    负责从配置文件加载RAG系统配置。
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """初始化RAG配置管理器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典（优先级高于config_path）
        """
        self.config = {}
        
        # 从文件加载配置
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # 从字典更新配置
        if config_dict:
            self.config.update(config_dict)
        
        logging.info("RAG配置管理器初始化完成")
    
    def __getattr__(self, name: str) -> Any:
        """属性访问器，允许通过点号访问配置项
        
        Args:
            name: 属性名
            
        Returns:
            配置项的值
        """
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def load_from_file(self, config_path: str):
        """从JSON文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            
            # 加载完整配置，保持原有结构以支持嵌套访问
            self.config = full_config
            logging.info(f"从 {config_path} 加载完整配置成功")
                
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            raise
    

    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置
        
        Returns:
            配置字典的深拷贝
        """
        import copy
        return copy.deepcopy(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置键，支持点分隔的嵌套键（如"wikipedia_data.chunk_size"）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"RAGConfig(config_keys={list(self.config.keys())})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"RAGConfig({self.config})"


def load_rag_config(config_path: Optional[str] = None, **kwargs) -> RAGConfig:
    """便捷函数：加载RAG配置
    
    Args:
        config_path: 配置文件路径
        **kwargs: 额外的配置参数
        
    Returns:
        RAG配置管理器实例
    """
    return RAGConfig(config_path=config_path, config_dict=kwargs)