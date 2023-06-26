# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/3 14:02
"""
    文件说明：
            
"""
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class GPT2DataSet(Dataset):
    """文档生成问题模型所需要的数据类"""
    def __init__(self, tokenizer, max_len, query_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据的最大长度
            query_max_len: 问题的最大长度
            data_dir: 保存缓存文件的路径
            data_set_name: 数据集名字
            path_file: 原始数据文件
            is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.query_max_len = query_max_len
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        """
        加载原始数据，生成数据处理后的数据
        Args:
            path_file: 原始数据路径
        Returns:
        """
        self.data_set = []
        # 遍历数据集
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                # 获取每个样本数据
                sample = json.loads(line.strip())
                # 使用convert_feature函数，对文档和问题进行索引化，生成模型所需数据格式
                input_ids, labels = self.convert_feature(sample)
                if input_ids is None:
                    continue
                self.data_set.append({"input_ids": input_ids, "labels": labels})
        return self.data_set

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含文档和问题，格式为{"content": content, "query": query}
        Returns:
        """
        # 对文档和问题进行tokenizer.tokenize分词
        content_tokens = self.tokenizer.tokenize(sample["content"])
        query_tokens = self.tokenizer.tokenize(sample["query"])
        # 当问题过长时，删除该条样本
        if len(query_tokens) > self.query_max_len:
            return None, None
        # 对文档进行裁剪
        if len(content_tokens) > self.max_len - len(query_tokens) - 3:
            content_tokens = content_tokens[:self.max_len - len(query_tokens) - 3]
        # 生成模型所需的input_ids和labels
        input_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(content_tokens) + [
            self.tokenizer.sep_token_id] + self.tokenizer.convert_tokens_to_ids(query_tokens) + [
                        self.tokenizer.sep_token_id]
        labels = input_ids
        assert len(input_ids) <= self.max_len
        assert len(input_ids) == len(labels)
        return input_ids, labels

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, labels_list = [], []
    for instance in batch_data:
        # 将input_ids和labels添加到对应的list中
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=0)}
