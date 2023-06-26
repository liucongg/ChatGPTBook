# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/28 22:37
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
    """摘要生成模型所需要的数据类"""

    def __init__(self, tokenizer, max_len, title_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据的最大长度
            title_max_len: 生成摘要的最大长度
            data_dir: 保存缓存文件的路径
            data_set_name: 数据集名字
            path_file: 原始数据文件
            is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        # space_id表示空格标记，由于一些标题中带有空格，如果直接使用tokenizer进行分词，会导致空格消失，会显得标题很奇怪
        # 但是又不方便同一替换成任意一个标点，因此将其用[unused11]替换。
        self.space_id = self.tokenizer.convert_tokens_to_ids("[unused11]")
        self.max_len = max_len
        self.title_max_len = title_max_len
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
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                # 使用convert_feature函数，对正文和摘要进行索引化，生成模型所需数据格式
                input_ids, token_type_ids = self.convert_feature(sample)
                self.data_set.append({"input_ids": input_ids, "mask": token_type_ids})
        return self.data_set

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含正文和摘要，格式为{"content": content, "title": title}
        Returns:
        """
        # 对正文进行tokenizer.tokenize分词
        content_tokens = self.tokenizer.tokenize(sample["content"])
        # 对摘要进行tokenizer.tokenize分词，注意tokenizer中已经将[unused11]作为一个分隔符，不会切割成多个字符
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ", "[unused11]"))
        # 判断如果摘要过长，进行截断
        if len(title_tokens) > self.title_max_len:
            title_tokens = title_tokens[:self.title_max_len]
        # 判断如果正文过长，进行截断
        if len(content_tokens) > self.max_len - len(title_tokens) - 3:
            content_tokens = content_tokens[:self.max_len - len(title_tokens) - 3]
        # 生成模型所需的input_ids和mask

        input_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(content_tokens) + [
            self.tokenizer.sep_token_id] + self.tokenizer.convert_tokens_to_ids(title_tokens) + [
                        self.tokenizer.sep_token_id]
        mask = [0] * (len(self.tokenizer.convert_tokens_to_ids(content_tokens)) + 1) + [1] * (
                len(self.tokenizer.convert_tokens_to_ids(title_tokens)) + 2)
        # 判断input_ids与mask长度是否一致
        assert len(input_ids) == len(mask)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.max_len
        return input_ids, mask

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
    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, mask_list = [], []
    for instance in batch_data:
        # 将input_ids和token_type_ids添加到对应的list中
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        mask_list.append(torch.tensor(instance["mask"], dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "mask": pad_sequence(mask_list, batch_first=True, padding_value=0)}
