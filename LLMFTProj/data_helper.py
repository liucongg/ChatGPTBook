# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/6 21:55
"""
    文件说明：
            
"""
import json
import random


def data_process(ori_path, train_path, test_path):
    """
    数据预处理
    Args:
        ori_path:
        train_path:
        test_path:

    Returns:

    """
    # 遍历原始文件
    data = []
    with open(ori_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            sample = json.loads(line.strip())
            # 从原始文件中抽取三元组内容，三元组内容之间用“_”连接
            spo_list = []
            for spo in sample["spo_list"]:
                spo_list.append("_".join([spo["h"]["name"], spo["relation"], spo["t"]["name"]]))
            # 多个三元组之间用“\n”连接
            data.append({"text": sample["text"], "spo": "\n".join(spo_list)})
    # 随机打乱数据集
    random.shuffle(data)

    fin_0 = open(train_path, "w", encoding="utf-8")
    fin_1 = open(test_path, "w", encoding="utf-8")
    for i, sample in enumerate(data):
        # 随机抽取50条数存入测试集中
        if i < 50:
            fin_1.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # 其余数据，存入训练集中
        else:
            fin_0.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin_0.close()
    fin_1.close()


if __name__ == '__main__':
    ori_path = "data/train.json"
    train_path = "data/spo_0.json"
    test_path = "data/spo_1.json"
    data_process(ori_path, train_path, test_path)
