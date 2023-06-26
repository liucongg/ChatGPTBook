# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/3 14:02
"""
    文件说明：
            
"""
import json
import random


def data_process(train_path, test_path, sft_save_path, rl_save_path, test_save_path):
    """数据处理，从原始阅读理解数据中抽取文本+问题数据"""

    content_ids = 0
    # 由于测试数据集不需要切分，所以直接遍历cmrc的测试数据集，保存文档和对应问题即可。
    fin = open(test_save_path, "w", encoding="utf-8")
    with open(test_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        for sample in data["data"]:
            for paras in sample["paragraphs"]:
                content = paras["context"]
                content_ids += 1
                for qas in paras["qas"]:
                    fin.write(json.dumps(
                        {"content": content, "query": qas["question"], "content_id": "content_{}".format(content_ids)},
                        ensure_ascii=False) + "\n")
    fin.close()
    # 由于训练数据集需要切分为SFT阶段使用和RL阶段使用，所以遍历cmrc的训练数据集时，需要记录文档ID。
    qg_data = []
    content_ids_set = set()
    with open(train_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        for sample in data["data"]:
            for paras in sample["paragraphs"]:
                content = paras["context"]
                content_ids += 1
                for qas in paras["qas"]:
                    qg_data.append(
                        {"content": content, "query": qas["question"], "content_id": "content_{}".format(content_ids)})
                    content_ids_set.add("content_{}".format(content_ids))

    content_ids_set = list(content_ids_set)
    # 将文档ID进行随机打乱，并均分
    random.shuffle(content_ids_set)
    sft_train_ids = content_ids_set[:int(len(content_ids_set) / 2)]
    print(sft_train_ids)
    ppo_train_ids = content_ids_set[int(len(content_ids_set) / 2):]
    print(ppo_train_ids)
    # 遍历所以文档-问题数据集，如果文档ID在SFT阶段中，则保存到SFT数据集中，否则保存到RL数据集中。
    fin_sft = open(sft_save_path, "w", encoding="utf-8")
    fin_rl = open(rl_save_path, "w", encoding="utf-8")
    for i, sample in enumerate(qg_data):
        if sample["content_id"] in sft_train_ids:
            fin_sft.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elif sample["content_id"] in ppo_train_ids:
            fin_rl.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin_sft.close()
    fin_rl.close()


if __name__ == '__main__':
    ori_train_path = "data/cmrc_train.json"
    ori_test_path = "data/cmrc_dev.json"
    sft_save_path = "data/sft_train.json"
    rl_save_path = "data/ppo_train.json"
    test_save_path = "data/test.json"
    data_process(ori_train_path, ori_test_path, sft_save_path, rl_save_path, test_save_path)
