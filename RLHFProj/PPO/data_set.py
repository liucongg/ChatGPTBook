# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/14 19:24
"""
    文件说明：
            
"""
import json
import random


class ExamplesSampler(object):
    """数据类"""
    def __init__(self, path):
        """初始化函数"""
        self.data = set()
        # 遍历数据
        with open(path, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                # 将文本内容加入到数据集合中，构建prompt数据集
                sample = json.loads(line.strip())
                self.data.add(sample["content"])
        self.data = list(self.data)

    def sample(self, n):
        """随机采样"""
        # 对数据集随机挑选n个prompt用于模型训练
        samples = random.sample(self.data, n)
        return samples

    def __len__(self):
        return len(self.data)
