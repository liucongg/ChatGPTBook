# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/14 17:40
"""
    文件说明：
            
"""

from utils import get_advantages_and_returns, actor_loss_function, critic_loss_function
from model import ActorModel, RewardModel, CriticModel
import argparse
from data_set import ExamplesSampler
import os
from transformers.models.bert import BertTokenizer
from torch.optim import Adam
import random
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def make_experience(args, actor_model, critic_model, ori_model, reward_model, input_ids, generate_kwargs):
    """获取经验数据"""
    actor_model.eval()
    critic_model.eval()
    with torch.no_grad():
        # 获取prompt内容长度
        prompt_length = input_ids.shape[1]
        # 使用动作模型通过已有提示生成指定内容，其中：seq_outputs为返回序列，包含prompt+生成的answer
        seq_outputs, attention_mask = actor_model.generate(input_ids, **generate_kwargs)
        # 通过动作模型和原始模型同时计算生成结果对应的log_probs
        action_log_probs = actor_model(seq_outputs, attention_mask)
        base_action_log_probs = ori_model(seq_outputs, attention_mask)
        # 通过评判模型计算生成的answer的分值
        value, _ = critic_model(seq_outputs, attention_mask, prompt_length)
        value = value[:, :-1]
        # 通过奖励模型计算生成奖励值，并对奖励值进行裁剪
        _, reward_score = reward_model.forward(seq_outputs, attention_mask, prompt_length=prompt_length)
        reward_clip = torch.clamp(reward_score, -args.reward_clip_eps, args.reward_clip_eps)
        # reward_clip = reward_score
        # 对动作模型和原始模型的log_probs进行kl散度计算，防止动作模型偏离原始模型
        kl_divergence = -args.kl_coef * (action_log_probs - base_action_log_probs)
        rewards = kl_divergence
        start_ids = input_ids.shape[1] - 1
        action_mask = attention_mask[:, 1:]
        ends_ids = start_ids + action_mask[:, start_ids:].sum(1)
        batch_size = action_log_probs.shape[0]
        # 将奖励值加到生成的answer最后一个token上
        for j in range(batch_size):
            rewards[j, start_ids:ends_ids[j]][-1] += reward_clip[j]
        # 通过奖励值计算优势函数
        advantages, returns = get_advantages_and_returns(value, rewards, start_ids, args.gamma, args.lam)

    experience = {"input_ids": input_ids, "seq_outputs": seq_outputs, "attention_mask": attention_mask,
                  "action_log_probs": action_log_probs, "value": value, "reward_score": reward_score,
                  "advantages": advantages, "returns": returns}
    return experience


def update_model(args, experience_list, actor_model, actor_optimizer, critic_model, critic_optimizer, tb_write,
                 ppo_step):
    """模型更新"""
    # 根据强化学习训练轮数，进行模型更新
    for _ in range(args.ppo_epoch):
        # 随机打乱经验池中的数据，并进行数据遍历
        random.shuffle(experience_list)
        for i_e, experience in enumerate(experience_list):
            ppo_step += 1
            start_ids = experience["input_ids"].size()[-1] - 1

            # 获取actor模型的log_probs
            action_log_probs = actor_model(experience["seq_outputs"], experience["attention_mask"])
            action_mask = experience["attention_mask"][:, 1:]
            # 计算actor模型损失值
            actor_loss = actor_loss_function(action_log_probs[:, start_ids:],
                                             experience["action_log_probs"][:, start_ids:], experience["advantages"],
                                             action_mask[:, start_ids:], args.policy_clip_eps)
            # actor模型梯度回传，梯度更新
            actor_loss.backward()
            tb_write.add_scalar("actor_loss", actor_loss.item(), ppo_step)
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), args.max_grad_norm)
            actor_optimizer.step()
            actor_optimizer.zero_grad()

            # 计算critic模型的value
            value, _ = critic_model(experience["seq_outputs"], experience["attention_mask"],
                                    experience["input_ids"].size()[-1])
            value = value[:, :-1]
            # 计算critic模型损失值
            critic_loss = critic_loss_function(value[:, start_ids:], experience["value"][:, start_ids:],
                                               experience["returns"], action_mask[:, start_ids:], args.value_clip_eps)
            # actor模型梯度回传，梯度更新
            critic_loss.backward()
            tb_write.add_scalar("critic_loss", critic_loss.item(), ppo_step)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), args.max_grad_norm)
            critic_optimizer.step()
            critic_optimizer.zero_grad()
    return ppo_step


def train(args, ori_model, actor_model, reward_model, critic_model, tokenizer, dataset, device, tb_write):
    """模型训练"""
    # 根据actor模型和critic模型构建actor优化器和critic优化器
    actor_optimizer = Adam(actor_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    critic_optimizer = Adam(critic_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    cnt_timesteps = 0
    ppo_step = 0
    experience_list = []
    mean_reward = []
    # 训练
    for i in range(args.num_episodes):
        for timestep in range(args.max_timesteps):
            cnt_timesteps += 1
            # 从数据集中随机抽取batch_size大小数据
            prompt_list = dataset.sample(args.batch_size)
            # 生成模型所需的input_ids
            input_ids = tokenizer.batch_encode_plus(prompt_list, return_tensors="pt",
                                                    max_length=args.max_len - args.query_len - 3,
                                                    truncation=True, padding=True)["input_ids"]
            input_ids = input_ids.to(device)
            generate_kwargs = {
                "min_length": -1,
                "max_length": input_ids.shape[1] + args.query_len,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "do_sample": args.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "num_return_sequences": args.num_return_sequences}
            # 生成经验数据，并添加到经验池中
            experience = make_experience(args, actor_model, critic_model, ori_model, reward_model, input_ids,
                                         generate_kwargs)
            experience_list.append(experience)
            # 记录数据中的奖励值
            mean_reward.extend(experience["reward_score"].detach().cpu().numpy().tolist())

            # 当到达更新步数，进行模型更新
            if (cnt_timesteps % args.update_timesteps == 0) and (cnt_timesteps != 0):
                # 打印并记录平均奖励值
                mr = np.mean(np.array(mean_reward))
                tb_write.add_scalar("mean_reward", mr, cnt_timesteps)
                print("mean_reward", mr)
                actor_model.train()
                critic_model.train()
                # 模型更新
                ppo_step = update_model(args, experience_list, actor_model, actor_optimizer, critic_model,
                                        critic_optimizer, tb_write, ppo_step)
                # 模型更新后，将经验池清空
                experience_list = []
                mean_reward = []
        # 模型保存
        actor_model.save_pretrained(os.path.join(args.output_dir, "checkpoint-{}".format(ppo_step)))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint-{}".format(ppo_step)))
        print("save model")


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--train_file_path', default='data/ppo_train.json',
                        type=str, help='训练数据集')
    parser.add_argument('--ori_model_path', default='sft_model/', type=str, help='SFT模型')
    parser.add_argument('--reward_model_path', default='rm_model/', type=str, help='奖励模型路径')
    parser.add_argument('--max_len', default=768, type=int, help='模型最大长度')
    parser.add_argument('--query_len', default=64, type=int, help='生成问题的最大长度')
    parser.add_argument('--batch_size', default=16, type=int, help='批次大小')
    parser.add_argument('--num_episodes', default=3, type=int, help='循环次数')
    parser.add_argument('--max_timesteps', default=80, type=int, help='单次训练最大步骤')
    parser.add_argument('--update_timesteps', default=20, type=int, help='模型更新步数')
    parser.add_argument('--kl_coef', default=0.02, type=float, help='kl散度概率')
    parser.add_argument('--ppo_epoch', default=2, type=int, help='强化学习训练轮数')
    parser.add_argument('--policy_clip_eps', default=0.2, type=float, help='策略裁剪')
    parser.add_argument('--value_clip_eps', default=0.2, type=float, help='值裁剪')
    parser.add_argument('--top_p', default=1.0, type=float, help='解码Top-p概率')
    parser.add_argument('--repetition_penalty', default=1.4, type=float, help='重复惩罚率')
    parser.add_argument('--do_sample', default=True, type=bool, help='随机解码')
    parser.add_argument('--num_return_sequences', default=1, type=int, help='生成内容个数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--reward_clip_eps', default=5.0, type=float, help='奖励值裁剪')
    parser.add_argument('--gamma', default=1.0, type=float, help='优势函数gamma值')
    parser.add_argument('--lam', default=0.95, type=float, help='优势函数lambda值')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率')
    parser.add_argument('--adam_epsilon', default=1e-5, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--output_dir', default="output_dir", type=str, help='模型保存路径')
    parser.add_argument('--seed', default=2048, type=int, help='')
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
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    tb_write = SummaryWriter()
    # 实例化原始模型、Actor模型、Reward模型和Critic模型
    ori_model = ActorModel(args.ori_model_path)
    ori_model.to(device)
    actor_model = ActorModel(args.ori_model_path)
    actor_model.to(device)

    reward_model = RewardModel.from_pretrained(args.reward_model_path)
    reward_model.to(device)

    critic_model = CriticModel.from_pretrained(args.reward_model_path)
    critic_model.to(device)

    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.ori_model_path, padding_side='left')
    tokenizer.eos_token_id = tokenizer.sep_token_id
    # 加载训练数据
    dataset = ExamplesSampler(args.train_file_path)
    print("数据量：{}".format(dataset.__len__()))
    # 开始训练
    train(args, ori_model, actor_model, reward_model, critic_model, tokenizer, dataset, device, tb_write)


if __name__ == '__main__':
    main()
