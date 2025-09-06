import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def read_csv(file_name):
    try:
        df = pd.read_csv(file_name)

        # 返回读取的DataFrame
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def load_df_from_tsv(path: Union[str, Path], sep="\t") -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep=sep,
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def load_tsv_to_dicts(path: Union[str, Path]) -> List[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        rows = [dict(e) for e in reader]
    return rows


def load_questions(filename="questions.csv"):
    """Loads csv of questions into a pandas dataframe"""

    questions = pd.read_csv(filename)
    questions.dropna(axis=1, how="all", inplace=True)  # drop all-null columns

    return questions


def save_questions(questions, filename="answers.csv"):
    """Saves dataframe of questions (with model answers) to csv"""

    questions.to_csv(filename, index=False)


def sample_dataset(dataset_name: str, data_path: str, sample_size: int, seed: int = 2025) -> List[Dict]:
    """
    从数据集中随机采样指定数量的数据
    
    Args:
        dataset_name: 数据集名称
        data_path: 数据文件路径
        sample_size: 采样数量
        seed: 随机种子，默认为2025
    
    Returns:
        采样后的数据列表
    """
    # 设置随机种子确保结果可重现
    random.seed(seed)
    
    print(f"正在从 {dataset_name} 数据集采样 {sample_size} 个样本...")
    
    # 检查文件是否存在
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载数据
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        raise RuntimeError(f"读取数据文件失败: {e}")
    
    print(f"原始数据集包含 {len(data)} 个样本")
    
    # 如果请求的样本数大于等于总数据量，返回全部数据
    if sample_size >= len(data):
        print(f"请求样本数 ({sample_size}) >= 总数据量 ({len(data)})，返回全部数据")
        return data
    
    # 随机采样
    sampled_data = random.sample(data, sample_size)
    print(f"✅ 成功采样 {len(sampled_data)} 个样本")
    
    return sampled_data