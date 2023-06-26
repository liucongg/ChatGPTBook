# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 杜振东.py
# @contact: zddu@iyunwen.com
# @time: 2023/4/1 18:45
"""
    文件说明：
            
"""
import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from data_set import ReviewQueryDataset, collate_func
from tqdm import tqdm, trange
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models(args, device):
    """
    加载模型
    Args:
        args: 训练参数配置信息
        device: 设备信息
    Returns:
    """
    actor_tokenizer = AutoTokenizer.from_pretrained(args.actor_model_name)
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.actor_model_name).to(device)
    ref_actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.actor_model_name)
    critic_tokenizer = BertTokenizer.from_pretrained(args.critic_model_name)
    critic_model = BertForSequenceClassification.from_pretrained(args.critic_model_name).to(device)
    return actor_model, ref_actor_model, actor_tokenizer, critic_model, critic_tokenizer


def init_ppo(args, actor_model, ref_actor_model, actor_tokenizer, dataset):
    """
    初始化PPO配置
    Args:
        args: 训练参数配置信息
        actor_model: 演员模型
        ref_actor_model: 参考模型
        actor_tokenizer: 演员模型分词器
        dataset: 构建好的数据集

    Returns:
    """
    config = PPOConfig(
        model_name=args.actor_model_name,
        learning_rate=args.learning_rate,
        ppo_epochs=args.num_train_epochs,
    )
    ppo_trainer = PPOTrainer(config, actor_model, ref_actor_model, actor_tokenizer, dataset=dataset['train'],
                             data_collator=collate_func)
    return ppo_trainer


def train(actor_tokenizer, critic_model, critic_tokenizer, device, ppo_trainer, args):
    """
    训练模型
    Args:
        actor_tokenizer: 演员模型分词器
        critic_model: 裁判模型
        critic_tokenizer： 裁判模型分词器
        device: 设备信息
        ppo_trainer: ppo训练器
        args: 训练参数配置信息
    Returns:
    """
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": actor_tokenizer.pad_token_id
    }
    # 定义输出样本长度函数
    output_size_sampler = LengthSampler(args.generate_min_len, args.generate_max_len)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']

        #### 从演员模型生成文本
        response_tensors = []
        for query in query_tensors:
            gen_len = output_size_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query.to(device), **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch['response'] = [actor_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### 计算情感得分
        texts = [q + r for q, r in zip(batch['query'], batch['response'])]
        encoded_inputs = critic_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        output = critic_model(**encoded_inputs)
        rewards = list(output.logits[:, 1].float())
        #### 运行PPO流程
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--actor_model_name', default="uer/gpt2-chinese-cluecorpussmall", type=str, help='预训练的演员模型名称')
    parser.add_argument('--critic_model_name', default="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment", type=str,
                        help='预训练的裁判模型名称')
    parser.add_argument('--data_dir', default='data/', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练时每个batch的大小')
    parser.add_argument('--learning_rate', default=1.41e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--generate_min_len', default=6, type=int, help='生成评价的最小长度')
    parser.add_argument('--generate_max_len', default=12, type=int, help='生成评价的最大长度')
    parser.add_argument('--output_dir', default='ppo_review_generate/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    rqd = ReviewQueryDataset()
    my_dataset = rqd.dataset
    # 实例化模型
    actor_model, ref_actor_model, actor_tokenizer, critic_model, critic_tokenizer = load_models(args, device)
    ppo_trainer = init_ppo(args, actor_model, ref_actor_model, actor_tokenizer, my_dataset)
    # 开始训练
    train(actor_tokenizer, critic_model, critic_tokenizer, device, ppo_trainer, args)
    actor_model.save_pretrained(args.output_dir)
    actor_tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
