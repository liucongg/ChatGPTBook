# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: generate_sample
# @author: 杜振东
# @contact: zddu@iyunwen.com
# @time: 2023/3/31 19:44
"""
    文件说明：生成偏正向情感评价内容示例样本
            
"""
import torch
import os
import argparse
from transformers import  AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import copy
import logging
from trl.core import LengthSampler
import pandas as pd
from data_set import ReviewQueryDataset
logger = logging.getLogger(__name__)
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--model_path', default='ppo_review_generate/', type=str, help='模型文件路径')
    parser.add_argument('--batch_size', default=4, type=int, help='生成评价的个数')
    parser.add_argument('--generate_min_len', default=6, type=int, help='生成评价的最小长度')
    parser.add_argument('--generate_max_len', default=12, type=int, help='生成评价的最大长度')
    parser.add_argument('--max_len', type=int, default=20, help='输入模型的最大长度')
    parser.add_argument('--eval_mode', type=int, default=0, help='验证模式,0:代表测试新生成，1:代表对比测试')
    return parser.parse_args()


def predict_one_sample(model, tokenizer, device, args, content, output_size_sampler, gen_kwargs):
    """
    对单个样本进行预测
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备信息
        args: 配置项信息
        content: 正文
        output_size_sampler: 输出样本长度函数
        gen_kwargs: 生成相关配置项
    Returns:
    """
    # 对正文进行预处理，并判断如果超长则进行截断
    if len(content) > args.max_len - 3:
        content = content[:args.max_len - 3]
    content_tokens = tokenizer.tokenize(content)
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    # 将input_ids进行扩充，扩充到需要预测摘要的个数，即batch_sizes
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    # 将input_ids变成tensor
    input_tensors = torch.tensor(input_ids).long().to(device)
    # 用于存放每一步解码的结果
    generated = []
    with torch.no_grad():
        # 遍历生成摘要最大长度
        output_size = output_size_sampler()
        outputs = model.generate(input_tensors,max_new_tokens=output_size, **gen_kwargs).squeeze()
        for output in outputs:
            generated.append(tokenizer.decode(output))
    return generated

def compare(model, ref_model, critic_model, tokenizer, critic_tokenizer, device, args, my_dataset, output_size_sampler, gen_kwargs):
    """
    基于训练前后模型效果评估
    Args:
        model: 生成模型
        ref_model: 参考原始模型
        critic_model: 观察者评估模型
        tokenizer: 生成模型分词器
        critic_tokenizer: 观察者模型分词器
        device: 设备信息
        args: 配置项信息
        my_dataset: 数据集
        output_size_sampler: 输出样本长度函数
        gen_kwargs: 生成相关配置项
    Returns:
    """
    input_data = dict()
    df_batch = my_dataset['train'][:].sample(args.batch_size)
    input_data['query'] = df_batch['query'].tolist()
    query_tensors = df_batch['input_ids'].tolist()
    response_tensors_ref, response_tensors = [], []
    #从现有模型与原始模型中续写评价
    for i in range(args.batch_size):
        gen_len = output_size_sampler()
        output = ref_model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
                                        max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
        response_tensors_ref.append(output)
        output = model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
                                    max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
        response_tensors.append(output)

    #对结果内容解码
    input_data['response (before)'] = [tokenizer.decode(response_tensors_ref[i]) for i in range(args.batch_size)]
    input_data['response (after)'] = [tokenizer.decode(response_tensors[i]) for i in range(args.batch_size)]

    #利用裁判模型对新老生成模型进行评估
    texts = [q + r for q,r in zip(input_data['query'], input_data['response (before)'])]
    encoded_inputs = critic_tokenizer(texts,padding=True,truncation=True,return_tensors='pt').to(device)
    output=critic_model(**encoded_inputs)
    rewards = list(output.logits[:,1].to('cpu').tolist())
    input_data['rewards (before)'] = rewards

    texts = [q + r for q,r in zip(input_data['query'], input_data['response (after)'])]
    encoded_inputs = critic_tokenizer(texts,padding=True,truncation=True,return_tensors='pt').to(device)
    output=critic_model(**encoded_inputs)
    rewards = list(output.logits[:,1].to('cpu').tolist())
    input_data['rewards (after)'] = rewards

    # 将结果保存在dataframe中
    df_results = pd.DataFrame(input_data)
    return df_results


def main():
    """主函数"""
    # 设置预测的配置参数
    args = set_args()
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    logger.info(device)
    # 定义输出样本长度函数
    output_size_sampler = LengthSampler(args.generate_min_len, args.generate_max_len)
    # 实例化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 定义生成相关配置项
    gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
    }
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    if args.eval_mode == 0:
        logger.info('开始对文本评价摘要，输入CTRL + C，则退出')
        while True:
            content = input("输入的评价前缀为：")
            titles = predict_one_sample(model, tokenizer, device, args, content, output_size_sampler, gen_kwargs)
            for i, title in enumerate(titles):
                print("生成的第{}个评价为：{}".format(i + 1, title))
    elif args.eval_mode == 1:
        rqd = ReviewQueryDataset()
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        ref_model.to(device)
        ref_model.eval()
        critic_tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
        critic_model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment').to(device)
        my_dataset = rqd.dataset
        my_dataset.set_format("pandas")
        df = compare(model, ref_model, critic_model, tokenizer, critic_tokenizer, device, args, my_dataset, output_size_sampler, gen_kwargs)
        print('df:',df)


if __name__ == '__main__':
    main()
