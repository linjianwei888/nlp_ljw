# -*- coding:utf-8 -*-
# @Time     : 2024/9/30 13:56
# @Author   : linjianwei
# @Email    : ai_ljw@163.com
# @File     : text_similarity_utils.py
# @Software : Pycharm

import torch
import torch.nn.functional as F

from libs.utils.open_embed import OpenEmbeddings

# 默认设备配置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from fuzzywuzzy import fuzz, process

import difflib
from typing import List, Dict, Optional, Tuple, Any


def edit_distance_similarity(text_a: str, text_b: str):
    """编辑距离相似度计算"""
    return fuzz.ratio(text_a, text_b) / 100


def batch_edit_distance_similarity(text: str, choices: List[str], limit: int = 3):
    """批量编辑距离相似度计算"""
    items = process.extract(query=text, choices=choices, limit=limit)
    return [(x[0], x[1] / 100) for x in items]


def find_most_similar(query: str, candidates: list[str]) -> str:
    """从候选列表中找到最相似的文本。"""
    return (
        difflib.get_close_matches(query, candidates, n=1, cutoff=0.0)[0]
        if candidates
        else None
    )


def find_most_similar_in_list_item(
    query: str, candidates: List[Dict], match_field: str
) -> Optional[Dict]:
    """
    从候选字典中找到最相似的文本。

    参数：
        query (str): 要匹配的文本。
        candidates (List[Dict]): 候选字典列表，每个字典包含一个文本字段。
        match_field (str): 字典中要匹配的字段名。

    返回：
        Optional[Dict]: 最相似的字典，如果没有找到则返回None。
    """
    if not candidates:
        return {}
    # 提取所有的text字段
    texts = [item.get(match_field, "") for item in candidates]
    # 找到最相似的文本
    matches = difflib.get_close_matches(query, texts, n=1, cutoff=0.0)
    if not matches:
        return {}
    best_text = matches[0]
    # 找到对应的原始字典
    for item in candidates:
        if item.get(match_field, "") == best_text:
            return item
    return {}  # 如果没有找到（理论上不会到这里）


def find_most_similar_in_list_item_vector(
    query: str, candidates: List[Dict], match_field: str, embedder: OpenEmbeddings
) -> Optional[Dict]:
    """
    使用向量相似度从候选字典中找到最相似的文本。

    参数：
        query (str): 要匹配的文本。
        candidates (List[Dict]): 候选字典列表。
        match_field (str): 字典中用于匹配的字段名。
        embedder (OpenEmbeddings): 用于生成文本嵌入的类。

    返回：
        Optional[Dict]: 最相似的字典，若无候选则返回空字典。
    """
    if not candidates:
        return {}

    texts = [item.get(match_field, "") for item in candidates]
    candidate_embeddings = embedder.embed_documents(texts)
    query_embedding = embedder.embed_documents([query])[0]

    # 转为 tensor
    candidate_tensor = torch.tensor(candidate_embeddings, dtype=torch.float32)
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)

    # 计算余弦相似度
    similarities = F.cosine_similarity(query_tensor, candidate_tensor)
    best_idx = torch.argmax(similarities).item()

    return candidates[best_idx]


def compute_topk_similarity(
    embedder, query_texts: List[str], candidate_texts: List[str], top_k: int = 1,
    batch_size: int = 32, normalize: bool = True, use_device: Optional[str] = None
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    优化的向量相似度计算函数，支持批处理、归一化和设备选择
    计算两组文本的向量余弦相似度，并返回每个query_texts与candidate_texts最相似的top_k分数和索引。

    参数:
        embedder: 支持embed_documents方法的嵌入模型
        query_texts (List[str]): 查询文本列表
        candidate_texts (List[str]): 候选文本列表
        top_k (int): 返回每个query_texts中与candidate_texts最相似的top_k
        batch_size (int): 批处理大小，用于处理大量文本时的内存优化
        normalize (bool): 是否对向量进行L2归一化（提升计算效率）
        use_device (Optional[str]): 指定使用的设备，None则使用全局device

    返回:
        top_k_scores (List[List[float]]): 每个query_texts与candidate_texts的top_k相似度分数（float）
        top_k_indices (List[List[int]]): 每个query_texts与candidate_texts的top_k索引（int）
    """
    if not query_texts or not candidate_texts:
        return [], []

    # 设备选择
    compute_device = device if use_device is None else torch.device(use_device)

    # 批量嵌入处理（避免一次性处理过多文本导致内存溢出）
    def batch_embed(texts: List[str], batch_size: int):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    # 获取嵌入向量
    if len(query_texts) > batch_size or len(candidate_texts) > batch_size:
        query_vecs = batch_embed(query_texts, batch_size)
        candidate_vecs = batch_embed(candidate_texts, batch_size)
    else:
        query_vecs = embedder.embed_documents(query_texts)
        candidate_vecs = embedder.embed_documents(candidate_texts)

    # 转换为张量并移至设备
    query_tensor = torch.tensor(query_vecs, dtype=torch.float32, device=compute_device)
    candidate_tensor = torch.tensor(candidate_vecs, dtype=torch.float32, device=compute_device)

    # L2归一化（使余弦相似度计算更高效）
    if normalize:
        query_tensor = F.normalize(query_tensor, p=2, dim=-1)
        candidate_tensor = F.normalize(candidate_tensor, p=2, dim=-1)
        # 归一化后，余弦相似度等价于点积
        sim_matrix = torch.mm(query_tensor, candidate_tensor.t())
    else:
        # 原始余弦相似度计算
        sim_matrix = F.cosine_similarity(
            query_tensor.unsqueeze(1), candidate_tensor.unsqueeze(0), dim=-1
        )

    # 计算top-k
    k = min(top_k, candidate_tensor.size(0))

    # 使用torch.topk的优化：当k较小时，sorted=False可以提升性能
    top_k_scores, top_k_indices = torch.topk(
        sim_matrix, k=k, dim=1, largest=True, sorted=True
    )

    # 批量转换为Python原生类型（更高效）
    if compute_device.type == 'cuda':
        # GPU到CPU的数据传输优化
        top_k_scores = top_k_scores.cpu()
        top_k_indices = top_k_indices.cpu()

    scores_list = top_k_scores.tolist()
    indices_list = top_k_indices.tolist()

    return scores_list, indices_list
