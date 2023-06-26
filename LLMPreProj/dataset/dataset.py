# -*- coding: utf-8 -*- 
# @Time : 2023/4/12 10:53 
# @Author : JunkRoy 
# @E-mail: shenroy92@gmail.com
# @File : dataset.py
import json
import tqdm
from torch.utils.data import Dataset
import torch
import os
import logging

logger = logging.getLogger(__name__)


def punctuation_standardization(string: str):
    """
    标点等内容进行规范化
    :param string:
    :return:
    """
    punctuation_dict = {"\u201c": "\"", "\u201d": "\"", "\u2019": "'", "\u2018": "'", "\u2013": "-"}
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


class PretrainDataset(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        :param tokenizer: 分词器
        :param max_len: 数据的最大长度
        :param data_dir: 保存缓存文件的路径
        :param data_set_name: 数据集名字
        :param path_file: 原始数据文件
        :param is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_set_name = data_set_name
        cached_feature_file = os.path.join(data_dir, "cached_{}_length_{}".format(data_set_name, max_len))

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
            samples = json.load(fh)

            for i in tqdm.trange(len(samples)):
                sample = samples[i]
                # if i > 1000:
                #     break
                input_ids, loss_masks, content = self.convert_feature(sample)
                sample["input_ids"] = input_ids
                sample["loss_mask"] = loss_masks
                sample["content"] = content
                self.data_set.append(sample)

        return self.data_set

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个字典
        Returns:
        """

        # 用空白字段占位prompt位置
        prompt = self.tokenizer.EncodeAsIds("").tokenization
        content = sample["title"] + sample["content"]
        content = punctuation_standardization(content)
        # 针对content进行tokenizer
        content_tokens = self.tokenizer.EncodeAsIds(content).tokenization
        tokens = [self.tokenizer.get_command('ENC').Id] + prompt + content_tokens
        # 设置loss_masks
        loss_masks = [0] * len(prompt) + [1] * len(tokens)
        # 控制长度满足最大长度
        input_ids = tokens[:self.max_len]
        loss_masks = loss_masks[:self.max_len]
        return input_ids, loss_masks, content

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance
