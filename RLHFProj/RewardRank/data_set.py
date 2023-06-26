# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/11 11:35
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


class RWDataSet(Dataset):
    """奖励模型所需要的数据类"""

    def __init__(self, tokenizer, max_len, query_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据的最大长度
            query_max_len: 生成query的最大长度
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
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                sample = json.loads(line.strip())
                # 使用convert_feature函数，对文本和问题进行索引化，生成模型所需数据格式
                input_ids, attention_mask = self.convert_feature(sample)
                if input_ids == None:
                    continue
                self.data_set.append({"input_ids": input_ids, "attention_mask": attention_mask})
        return self.data_set

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个list，包含多个样本，1个正样本和6个负样本
        Returns:
        """
        # 判断如果正样本的问题长度大于最大长度，丢弃该条数据
        if len(self.tokenizer.tokenize(sample[0]["answer"])) > self.query_max_len:
            return None, None

        input_ids_list, attention_mask_list = [], []

        for ism, s in enumerate(sample):
            # 对文本和问题进行分词，并根据最大长度进行裁剪
            content_tokens = self.tokenizer.tokenize(s["prompt"])
            query_tokens = self.tokenizer.tokenize(s["answer"])
            query_tokens = query_tokens[:self.query_max_len]
            content_max_len = self.max_len - len(query_tokens) - 3
            content_tokens = content_tokens[:content_max_len]
            # 生成模型所需的input_ids和mask
            input_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(content_tokens) + [
                self.tokenizer.sep_token_id] + self.tokenizer.convert_tokens_to_ids(query_tokens) + [
                            self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            # 将每个input_ids和mask加入到list当中，用于后续模型训练
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return input_ids_list, attention_mask_list

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
    input_ids_list, attention_mask_list = [], []

    for instance in batch_data:
        # 将input_ids和token_type_ids添加到对应的list中
        for input_ids, attention_mask in zip(instance["input_ids"], instance["attention_mask"]):
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}
