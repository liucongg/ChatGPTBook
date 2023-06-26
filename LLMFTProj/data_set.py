# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/4 14:42
"""
    文件说明：
            
"""
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        """
        初始化函数
        Args:
            data_path: 数据文件
            tokenizer: 分词器
            max_len: 模型输入最大长度
            max_src_len: 模型源文本最大长度
            prompt_text: 提示模板
        """
        # 根据模型最大长度和源文本最大长度计算目标文本的最大长度
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        # 遍历数据文件内容
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                # 利用json.loads进行数据加载
                sample = json.loads(line.strip())
                # 利用分词器，对源文本和提示模板进行分词
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)
                # 根据对分词后的源文本进行裁剪
                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]
                # 对目标文本进行分词，并根据目标文本最大长度进行裁剪
                tgt_tokens = tokenizer.tokenize(sample["answer"])
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                # 将各项分词结果进行拼接，并转换成模型所需的索引ID格式
                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # 对于训练模型的标签，仅保留目标文本索引ID，其他内容设置成-100，模型不计算对应的损失
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                # 对最终结果进行填充，填充到模型的最大长度
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                # 将每个样本进行保存，用于后续训练使用
                self.all_data.append(
                    {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, labels_list = [], []
    for instance in batch:
        # 将input_ids和token_type_ids添加到对应的list中
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=20003),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=20003)}

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")