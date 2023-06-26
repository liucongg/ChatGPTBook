# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/3/20 23:05
"""
    文件说明：
            
"""
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import logging

logger = logging.getLogger(__name__)


class PromptDataSet(Dataset):
    """Prompt数据类"""

    def __init__(self, tokenizer, max_len, template, label_dict, data_dir, data_set_name, path_file=None,
                 is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据最大长度
            template: prompt模板
            label_dict: 答案空间映射字典
            data_dir: 数据保存路径
            data_set_name: 数据集名称
            path_file: 原始数据文件
            is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = template
        self.label_dict = label_dict
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

        # 根据答案空间映射字典获取答案字典id和对应mask向量
        self.words_ids, self.words_ids_mask = self.get_verbalizer()

    def get_verbalizer(self):
        """
        根据答案空间映射字典获取答案字典id和对应mask向量
        Returns:

        """
        # 获取标签词
        label_words = []
        for label, verbalizer in self.label_dict.items():
            label_words.append(verbalizer["label_words"])

        all_ids = []
        # 遍历每个标签的标签词，构建标签词id列表
        for words_per_label in label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        # 判断单个标签词最大长度
        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        # 判断每个类别最大标签词个数
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        # 获取标签词id列表对应的掩码列表
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        # 将标签词id列表进行填充
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]
        # 返回标签词id及掩码矩阵，用于答案空间映射
        return torch.tensor(words_ids), torch.tensor(words_ids_mask)

    def load_data(self, path_file):
        """
        加载原始数据，生成数据处理后的数据
        Args:
            path_file: 原始数据路径
        Returns:
        """
        data_set = []
        # 遍历数据文件
        with open(path_file, "r", encoding="utf-8") as fh:
            for _, line in enumerate(fh):
                sample = json.loads(line.strip())
                # 对每个评论数据进行prompt构建，并获取模型输入所需内容，以及mask填充位置
                input_ids, attention_mask, mask_index = self.convert_feature(sample["text"])
                # 获取每个评论数据的标签id
                label = self.label_dict[sample["label"]]["label_id"]
                # 将所有数据添加到data_set中，待后续使用
                data_set.append({"input_ids": input_ids, "attention_mask": attention_mask, "mask_index": mask_index,
                                 "label": label, "text": sample["text"]})
        return data_set

    def convert_feature(self, text):
        """
        数据处理函数
        Args:
            text: 评论文本数据
        Returns:
        """
        # 当评论数据过长时，进行切断操作
        if len(text) > self.max_len - len(self.template):
            text = text[:self.max_len - len(self.template)]
        # 将prompt模板与评论数据融合
        text = "[CLS]" + self.template.replace("{mask}", "[MASK]").replace("{text}", text) + "[SEP]"
        # 对数据进行tokenize分词
        text_tokens = self.tokenizer.tokenize(text)
        # 获取[MASK]位置，用于填词
        try:
            mask_index = text_tokens.index("[MASK]")
        except:
            raise Exception("模板中缺少待填充的{mask}字段")
        # 生成模型训练所需的input_ids和attention_mask
        input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = [1] * len(input_ids)
        # 数据验证，判断数据是否小于最大长度
        assert len(input_ids) <= self.max_len
        return input_ids, attention_mask, mask_index

    def __len__(self):
        """获取数据总长度"""
        return len(self.data_set)

    def __getitem__(self, idx):
        """获取每个实例数据"""
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
    input_ids_list, attention_mask_list = [], []
    mask_index_list, label_list = [], []
    for instance in batch_data:
        # 将input_ids、token_type_ids、mask_index和label添加到对应的list中
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        attention_mask_list.append(torch.tensor(instance["attention_mask"], dtype=torch.long))
        mask_index_list.append(instance["mask_index"])
        label_list.append(instance["label"])
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "mask_index": torch.tensor(mask_index_list, dtype=torch.long),
            "label": torch.tensor(label_list, dtype=torch.long)}
