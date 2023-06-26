# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/3 15:00
"""
    文件说明：
            
"""
from transformers import BertTokenizer, GPT2LMHeadModel
import torch
import os
import argparse


def predict_one_sample(tokenizer, model, content, device):
    """
    对单个样本进行预测
    Args:
        tokenizer: 分词器
        model: 模型
        content: 正文
        device: 设备信息
    Returns:
    """
    # 设置模型生成内容配置信息
    generation_kwargs = {"min_length": 3,
                         "max_new_tokens": 64,
                         "top_p": 0.9,
                         "repetition_penalty": 1.2,
                         "do_sample": True,
                         "num_return_sequences": 2,
                         "pad_token_id": tokenizer.sep_token_id,
                         "eos_token_id": tokenizer.eos_token_id,
                         }
    # 对文本内容进行编码
    content = tokenizer.encode(content, max_length=768-64, return_tensors="pt", truncation=True).to(device)
    # 生成结果
    response = model.generate(content, **generation_kwargs)
    # 生成内容去除原始文本内容，获取问题内容
    response = response[:, content.shape[1]:]
    # 对每个问题内容进行解码，获取问题字符串
    query = [tokenizer.decode(r.squeeze()).replace(" ", "").replace("[SEP]", "").replace("##", "").replace("[UNK]", "")
             for r in response]
    return query


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设备信息')
    parser.add_argument('--model_path', default="sft_model", type=str, help='训练完成的模型路径')
    return parser.parse_args()


def main():
    """主函数"""
    # 设置预测的配置参数
    args = set_args()
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 实例化tokenizer和model
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tokenizer.eos_token = "[SEP]"
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print('开始对文本生成问题，输入CTRL + C，则退出')

    while True:
        content = input("输入的正文为：")
        querys = predict_one_sample(tokenizer, model, content, device)
        for i, query in enumerate(querys):
            print("生成的第{}个问题为：{}".format(i + 1, query))

if __name__ == '__main__':
    main()
