# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/28 22:58
"""
    文件说明：
            
"""
import json
import os
import random


def data_process(path, save_train_path, save_test_path):
    """

    Args:
        path:
        save_train_path:
        save_test_path:

    Returns:

    """
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            line = line.strip().split("\t")
            sample = {"content": line[1], "title": line[0]}
            data.append(sample)

    random.shuffle(data)

    fin_train = open(save_train_path, "w", encoding="utf-8")
    json.dump(data[:-2000], fin_train, ensure_ascii=False, indent=4)
    fin_train.close()

    fin_test = open(save_test_path, "w", encoding="utf-8")
    json.dump(data[-2000:], fin_test, ensure_ascii=False, indent=4)
    fin_test.close()


if __name__ == '__main__':
    path = "data/csl_40k.tsv"
    train_path = "data/train.json"
    test_path = "data/test.json"
    data_process(path, train_path, test_path)
