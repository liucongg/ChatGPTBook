# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 杜振东.py
# @contact: zddu@iyunwen.com
# @time: 2023/3/28 20:17
"""
    文件说明：
            
"""
from transformers import BertTokenizer
from datasets import load_dataset
from trl.core import LengthSampler
import logging

logger = logging.getLogger(__name__)


class ReviewQueryDataset():
    """摘要生成模型所需要的数据类"""

    def __init__(self, tokenizer_name="uer/gpt2-chinese-cluecorpussmall", dataset_name='amazon_reviews_multi',
                 max_len=50, query_min_len=2, query_max_len=8):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            dataset_name: 数据集名字
            max_len: 数据的最大长度
            query_min_len: 生成评价输入的最小长度
            query_max_len: 生成评价输入的最大长度
        """
        self.query_min_len = query_min_len
        self.query_max_len = query_max_len
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_len = max_len
        self.input_size = LengthSampler(self.query_min_len, self.query_max_len)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        logger.info('load dataset...')
        self.dataset = load_dataset(path=self.dataset_name, name="zh")
        self.preprocess()

    def tokenize(self, input):
        """
        加载原始数据，完成原始样本转换
        即将原始样本抽取前【min,max】个Token
        默认为【2-8】个Token
        Args:
            input: 原始数据样本

        Returns:
            input: 增加input_ids和query字段的样本
        """
        input["input_ids"] = self.tokenizer.encode(input["review_body"][:self.input_size()])
        input["query"] = self.tokenizer.decode(input["input_ids"])
        return input

    def preprocess(self):
        """
        对数据集原始文本开展数据预处理：
        -1.过滤过短文本【低于50字符评论】
        -2.保留原始评价前【min,max】个样本
        Returns:
        """
        logger.info('start preprocess...')
        self.dataset = self.dataset.filter(lambda x: len(x["review_body"]) > self.max_len, batched=False)
        self.dataset = self.dataset.map(self.tokenize, batched=False)
        self.dataset.set_format(type='torch')
        logger.info('preprocess finish!')


def collate_func(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成所需形式
    Args:
        batch_data: batch数据
    Returns:
    """
    return dict((key, [d[key] for d in batch_data]) for key in batch_data[0])
