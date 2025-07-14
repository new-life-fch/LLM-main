"""
知识提取组件
从Wikidata等知识源提取高质量的知识三元组
"""

import json
import logging
import requests
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd


class KnowledgeExtractor(ABC):
    """知识提取器基类"""
    
    @abstractmethod
    def extract_triplets(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """提取知识三元组"""
        pass
    
    @abstractmethod
    def get_high_frequency_relations(self, top_k: int = 1000) -> List[str]:
        """获取高频关系类型"""
        pass


class WikidataExtractor(KnowledgeExtractor):
    """
    Wikidata知识提取器
    通过SPARQL查询API提取高质量的知识三元组
    """
    
    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",
        cache_dir: str = "./cache/wikidata",
        rate_limit: float = 1.0  # 请求间隔（秒）
    ):
        """
        初始化Wikidata提取器
        
        Args:
            endpoint: SPARQL查询端点
            cache_dir: 缓存目录
            rate_limit: 请求速率限制
        """
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # 预定义的高质量关系
        self.high_quality_relations = [
            # 基本信息
            "P31",   # instance of
            "P279",  # subclass of
            "P106",  # occupation
            "P27",   # country of citizenship
            "P19",   # place of birth
            "P20",   # place of death
            "P569",  # date of birth
            "P570",  # date of death
            
            # 地理位置
            "P17",   # country
            "P131",  # located in administrative territorial entity
            "P36",   # capital
            "P625",  # coordinate location
            "P2044", # elevation above sea level
            
            # 创作和发明
            "P50",   # author
            "P57",   # director
            "P175",  # performer
            "P86",   # composer
            "P84",   # architect
            "P61",   # discoverer or inventor
            "P577",  # publication date
            
            # 组织关系
            "P108",  # employer
            "P69",   # educated at
            "P463",  # member of
            "P488",  # chairperson
            "P35",   # head of state
            "P6",    # head of government
            
            # 家庭关系
            "P22",   # father
            "P25",   # mother
            "P26",   # spouse
            "P40",   # child
            "P3373", # sibling
            
            # 奖项和成就
            "P166",  # award received
            "P102",  # member of political party
            "P39",   # position held
            
            # 科学和技术
            "P138",  # named after
            "P1441", # present in work
            "P179",  # part of the series
            "P921",  # main subject
        ]
        
        logging.info(f"Wikidata提取器初始化完成，缓存目录: {cache_dir}")
    
    def _rate_limit_wait(self):
        """等待速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行SPARQL查询
        
        Args:
            query: SPARQL查询语句
            
        Returns:
            查询结果列表
        """
        self._rate_limit_wait()
        
        headers = {
            'User-Agent': 'CausalEditor/1.0 (https://github.com/causal-editor)',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', {}).get('bindings', [])
            
        except Exception as e:
            logging.error(f"SPARQL查询失败: {e}")
            return []
    
    def extract_triplets(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        提取知识三元组
        
        Args:
            limit: 提取数量限制
            
        Returns:
            知识三元组列表
        """
        cache_file = self.cache_dir / f"triplets_{limit}.json"
        
        # 检查缓存
        if cache_file.exists():
            logging.info(f"从缓存加载三元组: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        triplets = []
        processed = set()
        
        for relation in self.high_quality_relations[:10]:  # 限制关系数量以避免过载
            relation_triplets = self._extract_triplets_for_relation(
                relation, limit=limit//len(self.high_quality_relations[:10])
            )
            
            for triplet in relation_triplets:
                triplet_key = f"{triplet['subject']}_{triplet['relation']}_{triplet['object']}"
                if triplet_key not in processed:
                    triplets.append(triplet)
                    processed.add(triplet_key)
                    
                if len(triplets) >= limit:
                    break
            
            if len(triplets) >= limit:
                break
        
        # 缓存结果
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)
        
        logging.info(f"提取了 {len(triplets)} 个知识三元组")
        return triplets
    
    def _extract_triplets_for_relation(self, relation_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        为特定关系提取三元组
        
        Args:
            relation_id: Wikidata关系ID
            limit: 数量限制
            
        Returns:
            该关系的三元组列表
        """
        # 构建SPARQL查询
        query = f"""
        SELECT ?subject ?subjectLabel ?object ?objectLabel WHERE {{
            ?subject wdt:{relation_id} ?object .
            ?subject rdfs:label ?subjectLabel .
            ?object rdfs:label ?objectLabel .
            FILTER(LANG(?subjectLabel) = "en")
            FILTER(LANG(?objectLabel) = "en")
            # 过滤出有足够链接的实体（质量控制）
            ?subject wikibase:sitelinks ?subjectSitelinks .
            FILTER(?subjectSitelinks > 5)
        }}
        LIMIT {limit}
        """
        
        results = self._sparql_query(query)
        triplets = []
        
        for result in results:
            try:
                subject_uri = result['subject']['value']
                subject_label = result['subjectLabel']['value']
                object_uri = result['object']['value'] 
                object_label = result['objectLabel']['value']
                
                # 提取实体ID
                subject_id = subject_uri.split('/')[-1]
                object_id = object_uri.split('/')[-1] if object_uri.startswith('http') else object_label
                
                triplet = {
                    'subject': subject_label,
                    'subject_id': subject_id,
                    'relation': relation_id,
                    'relation_label': self._get_relation_label(relation_id),
                    'object': object_label,
                    'object_id': object_id,
                    'confidence': 1.0,  # Wikidata数据默认高置信度
                    'source': 'wikidata'
                }
                
                # 生成自然语言文本
                triplet['text'] = self._generate_text(triplet)
                
                triplets.append(triplet)
                
            except Exception as e:
                logging.warning(f"解析三元组失败: {e}")
                continue
        
        return triplets
    
    def _get_relation_label(self, relation_id: str) -> str:
        """获取关系的人类可读标签"""
        relation_labels = {
            "P31": "is a",
            "P279": "is a subclass of", 
            "P106": "has occupation",
            "P27": "is a citizen of",
            "P19": "was born in",
            "P20": "died in",
            "P569": "was born on",
            "P570": "died on",
            "P17": "is located in country",
            "P131": "is located in",
            "P36": "has capital",
            "P625": "is located at coordinates",
            "P50": "was written by",
            "P57": "was directed by",
            "P175": "was performed by",
            "P86": "was composed by",
            "P577": "was published on",
            "P108": "works for",
            "P69": "was educated at",
            "P463": "is a member of",
            "P22": "has father",
            "P25": "has mother", 
            "P26": "is married to",
            "P40": "has child",
            "P166": "received award",
        }
        return relation_labels.get(relation_id, relation_id)
    
    def _generate_text(self, triplet: Dict[str, Any]) -> str:
        """
        为三元组生成自然语言文本
        
        Args:
            triplet: 知识三元组
            
        Returns:
            自然语言文本
        """
        subject = triplet['subject']
        relation = triplet['relation_label']
        obj = triplet['object']
        
        # 根据关系类型生成不同的文本模板
        if 'born' in relation:
            if 'on' in relation:
                return f"{subject} was born on {obj}."
            else:
                return f"{subject} was born in {obj}."
        elif 'died' in relation:
            if 'on' in relation:
                return f"{subject} died on {obj}."
            else:
                return f"{subject} died in {obj}."
        elif 'capital' in relation:
            return f"The capital of {subject} is {obj}."
        elif 'citizen' in relation:
            return f"{subject} is a citizen of {obj}."
        elif 'occupation' in relation:
            return f"{subject} is a {obj}."
        elif 'written by' in relation:
            return f"{subject} was written by {obj}."
        elif 'directed by' in relation:
            return f"{subject} was directed by {obj}."
        else:
            return f"{subject} {relation} {obj}."
    
    def get_high_frequency_relations(self, top_k: int = 1000) -> List[str]:
        """
        获取高频关系类型
        
        Args:
            top_k: 返回的关系数量
            
        Returns:
            高频关系ID列表
        """
        return self.high_quality_relations[:top_k]
    
    def get_popular_entities(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        获取热门实体
        
        Args:
            limit: 数量限制
            
        Returns:
            热门实体列表
        """
        cache_file = self.cache_dir / f"popular_entities_{limit}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 查询有最多外链的实体
        query = f"""
        SELECT ?entity ?entityLabel ?sitelinks WHERE {{
            ?entity wikibase:sitelinks ?sitelinks .
            ?entity rdfs:label ?entityLabel .
            FILTER(LANG(?entityLabel) = "en")
            FILTER(?sitelinks > 10)
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT {limit}
        """
        
        results = self._sparql_query(query)
        entities = []
        
        for result in results:
            try:
                entity_uri = result['entity']['value']
                entity_label = result['entityLabel']['value']
                sitelinks = int(result['sitelinks']['value'])
                
                entity_id = entity_uri.split('/')[-1]
                
                entities.append({
                    'id': entity_id,
                    'label': entity_label,
                    'sitelinks': sitelinks,
                    'uri': entity_uri
                })
                
            except Exception as e:
                logging.warning(f"解析实体失败: {e}")
                continue
        
        # 缓存结果
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        logging.info(f"获取了 {len(entities)} 个热门实体")
        return entities


class CSVKnowledgeExtractor(KnowledgeExtractor):
    """
    从CSV文件提取知识的提取器
    适用于已有的结构化知识数据
    """
    
    def __init__(self, csv_path: str):
        """
        初始化CSV知识提取器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        logging.info(f"CSV知识提取器初始化完成，数据量: {len(self.df)}")
    
    def extract_triplets(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        从CSV提取三元组
        
        期望CSV格式包含列: subject, relation, object, text (可选)
        """
        triplets = []
        
        for _, row in self.df.head(limit).iterrows():
            try:
                triplet = {
                    'subject': str(row['subject']),
                    'relation': str(row['relation']),
                    'object': str(row['object']),
                    'text': str(row.get('text', f"{row['subject']} {row['relation']} {row['object']}.")),
                    'confidence': float(row.get('confidence', 1.0)),
                    'source': 'csv'
                }
                triplets.append(triplet)
                
            except Exception as e:
                logging.warning(f"解析CSV行失败: {e}")
                continue
        
        return triplets
    
    def get_high_frequency_relations(self, top_k: int = 1000) -> List[str]:
        """获取高频关系"""
        relation_counts = self.df['relation'].value_counts()
        return relation_counts.head(top_k).index.tolist() 