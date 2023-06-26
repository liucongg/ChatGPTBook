# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/3/21 13:36
"""
    文件说明：
            
"""
import json
import os
import random
import pandas as pd


def data_process(path, save_train_path, save_test_path):
    """
    数据预处理
    Args:
        path: 原始酒店评价情感数据文件
        save_train_path: 保存训练文件
        save_test_path: 保存测试文件

    Returns:

    """
    data = []
    # 读取csv文件
    df = pd.read_csv(path)
    # 遍历文件每一行
    for i, row in df.iterrows():
        if row["label"] == 1:
            label = "正向"
        else:
            label = "负向"
        # 评价数据过长的过滤
        if len(row["review"]) > 512:
            continue
        data.append({"label": label, "text": row["review"]})
    # 数据进行随机打乱
    random.shuffle(data)

    fin_train = open(save_train_path, "w", encoding="utf-8")
    fin_test = open(save_test_path, "w", encoding="utf-8")
    pos_n = 0
    neg_n = 0
    for sample in data:
        # 在训练集中保存20个正样本数据
        if pos_n < 20 and sample["label"] == "正向":
            fin_train.write(json.dumps(sample, ensure_ascii=False) + "\n")
            pos_n += 1
        # 在训练集中保存20个负样本数据
        elif neg_n < 20 and sample["label"] == "负向":
            fin_train.write(json.dumps(sample, ensure_ascii=False) + "\n")
            neg_n += 1
        # 其他数据保存到测试集中
        else:
            fin_test.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin_train.close()
    fin_test.close()


if __name__ == '__main__':
    path = "data/ChnSentiCorp_htl_all.csv"
    train_path = "data/train.json"
    test_path = "data/test.json"
    data_process(path, train_path, test_path)
    # https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all
