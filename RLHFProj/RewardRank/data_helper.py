# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/11 11:03
"""
    文件说明：
            
"""
import json
import re
import random
from collections import defaultdict
import os
import torch
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel


def seg_content_to_sentences(content):
    """
    将文本内容进行分句
    Args:
        content: 文本内容

    Returns:

    """
    # 将文本内容按照问号、句号、感叹号进行句子切分
    sentences = re.split(r"([?？。！!-])", content)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    if sentences[-1] == "":
        sentences.pop(-1)
    return sentences


def data_process(path, save_path, vocab_path, model, tokenizer, device):
    """数据处理"""
    # 加载词典，用于随机query构建
    vocab_list = []
    with open(vocab_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if "[" in line or "##" in line:
                continue
            vocab_list.append(line)

    # 遍历数据集，按照文本和问题构建字典，一个文本内容对应多个问题
    content_query_dict = defaultdict(list)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        for sample in data["data"]:
            for paras in sample["paragraphs"]:
                content = paras["context"]
                for qas in paras["qas"]:
                    content_query_dict[content].append(
                        {"query": qas["question"], "answer_text": qas["answers"][0]["text"]})

    # 如果是训练集，则随机采样两次，实现动态负样本构造
    if "train" in save_path:
        copy_number = 2
    else:
        copy_number = 1

    all_data = []
    for _ in range(copy_number):
        # 遍历文本数据
        content_list = list(content_query_dict.keys())
        for i, content in enumerate(content_list):
            for sample in content_query_dict[content]:
                # 构造正样本
                pos_sample = {"prompt": content, "answer": sample["query"]}
                # 构造负样本1-模型生成的query
                generation_kwargs = {"min_length": -1,
                                     "max_new_tokens": 64,
                                     "top_k": 20,
                                     "top_p": 1.0,
                                     "repetition_penalty": 1.2,
                                     "do_sample": True,
                                     "num_return_sequences": 1,
                                     "pad_token_id": 0,
                                     "eos_token_id": 102,
                                     }
                input_ids = tokenizer.encode(content, max_length=900, return_tensors="pt", truncation=True).to(device)
                response = model.generate(input_ids, **generation_kwargs)
                response = response[:, input_ids.shape[1]:]
                query = [tokenizer.decode(r.squeeze()).replace(" ", "").replace("[SEP]", "").replace("##", "").replace(
                    "[UNK]", "") for r in response][0]
                neg_sample_1 = {"prompt": content, "answer": query}
                # 构造负样本2-在其他文本中随机挑选一个query
                temp_content = random.choice(content_list)
                while temp_content == content:
                    temp_content = random.choice(content_list)
                neg_sample_2 = {"prompt": content, "answer": random.choice(content_query_dict[temp_content])["query"]}
                # 构造负样本3-文本中随机选取一句话
                neg_sample_3 = {"prompt": content, "answer": random.choice(seg_content_to_sentences(content))}
                # 构造负样本4-不完整的query，对query进行裁剪、或增加额外字符
                query_len = len(sample["query"])
                start, end = min(3, int(query_len / 3)), min(3, int(query_len / 3))
                if start == end:
                    answer = sample["query"][:3]
                else:
                    answer = sample["query"][:random.choice(list(range(start, end)))]
                if random.random() < 0.5:
                    answer += "？"
                else:
                    answer += "".join(random.sample(vocab_list, random.choice(range(5, 10))))
                neg_sample_4 = {"prompt": content, "answer": answer}
                # 构造负样本5-空内容，无query
                neg_sample_5 = {"prompt": content, "answer": ""}
                # 构造负样本6-通过字典随机选取字符生成query
                if random.random() < 0.8:
                    answer = "".join(random.sample(vocab_list, random.choice(range(6, 30))))
                else:
                    answer = "".join(random.choice(vocab_list) * random.choice(range(6, 30)))
                neg_sample_6 = {"prompt": content, "answer": answer}
                # 将数据添加到all_data中，每个正样本对应6个负样本
                all_data.append(
                    [pos_sample, neg_sample_1, neg_sample_2, neg_sample_3, neg_sample_4, neg_sample_5, neg_sample_6])

    random.shuffle(all_data)
    # 将数据进行保存
    fin = open(save_path, "w", encoding="utf-8")
    for sample in all_data:
        fin.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin.close()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("pretrain_model/")
    tokenizer.eos_token = "[SEP]"
    model = GPT2LMHeadModel.from_pretrained("pretrain_model/")
    model.to(device)
    model.eval()
    # 训练集数据构造
    vocab_path = "pretrain_model/vocab.txt"
    train_path = "data/cmrc_train.json"
    save_train_path = "data/train.json"
    data_process(train_path, save_train_path, vocab_path, model, tokenizer, device)
    # 测试集数据构造
    test_path = "data/cmrc_dev.json"
    save_test_path = "data/test.json"
    data_process(test_path, save_test_path, vocab_path, model, tokenizer, device)
