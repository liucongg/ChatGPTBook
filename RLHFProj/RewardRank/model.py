# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: model_2
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/14 9:35
"""
    文件说明：
            
"""
import torch
from transformers.models.gpt2 import GPT2Model, GPT2PreTrainedModel


class RewardModel(GPT2PreTrainedModel):
    """奖励模型"""

    def __init__(self, config):
        """
        初始化函数
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout_fn = torch.nn.Dropout()
        self.value_fn = torch.nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        """
        前向函数
        Args:
            input_ids: 输入序列在词表中的索引序列，size: [batch_size, sequence_length]
            attention_mask: 掩码序列，size: [batch_size, sequence_length]，一般情况下，与input_ids相同

        Returns:

        """
        # 获取GPT2模型的输出结果
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        # 对hidden_states进行dropout
        hidden_states = self.dropout_fn(hidden_states)
        # 获取每个token的预测值
        values = self.value_fn(hidden_states).squeeze(-1)
        # 获取批次个数和数据最大长度
        true_bs, seq_len = input_ids.shape[0], input_ids.shape[1]
        # 获取真实批次，并对input_ids、values进行转换
        bs = int(true_bs / 7)
        values = values.reshape([bs, 7, seq_len])
        input_ids = input_ids.reshape([bs, 7, seq_len])
        loss, add_count = 0.0, 0
        # 遍历每个批次内容
        for ibs in range(bs):
            rank_reward = values[ibs, :, :]
            input_ids_ = input_ids[ibs, :, :]
            # 对于一个批次内的样本进行遍历，需要预测概率值，正样本>负样1>负样2>负样3>负样4>负样5>负样6
            for i in range(len(rank_reward) - 1):
                for j in range(i + 1, len(rank_reward)):
                    # 前一个样本要比后一个样本更好
                    one_rank_reward = rank_reward[i, :]
                    one_input_ids = input_ids_[i, :]
                    two_rank_reward = rank_reward[j, :]
                    two_input_ids = input_ids_[j, :]
                    # 找到两个样本差异的地方，即生成query的差异内容
                    check_divergence = (one_input_ids != two_input_ids).nonzero()
                    one_inds = (one_input_ids == 0).nonzero()
                    one_ind = one_inds[0].item() if len(one_inds) > 0 else seq_len
                    if len(check_divergence) == 0:
                        end_ind = two_rank_reward.size(-1)
                        divergence_ind = end_ind - 1
                    else:
                        two_inds = (two_input_ids == 0).nonzero()
                        two_ind = two_inds[0].item() if len(two_inds) > 0 else seq_len
                        end_ind = max(one_ind, two_ind)
                        divergence_ind = check_divergence[0]
                    # 获取差异内容的value值
                    one_truncated_reward = one_rank_reward[divergence_ind:end_ind]
                    two_truncated_reward = two_rank_reward[divergence_ind:end_ind]
                    # 计算loss，并进行累计
                    loss += -torch.log(torch.sigmoid(one_truncated_reward - two_truncated_reward)).mean()
                    add_count += 1
        # 计算一个批次内的loss平均值
        loss = loss / add_count
        return loss, add_count

    def predict(self, input_ids, attention_mask, prompt_length):
        """对单条样本预测"""
        # 获取GPT2模型输出
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)[0]
        # 将每个token获取对应的value
        values = self.value_fn(hidden_states).squeeze(-1)
        # 获取除prompt部分，最后一个非pad字符value最为奖励值
        bs, seq_len = input_ids.shape[0], input_ids.shape[1]
        scores = []
        # 遍历batch中每个结果
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            # 获取生成答案部分pad的全部索引
            idxs = (input_id[prompt_length:] == 0).nonzero()
            # 如果没有pad，那么奖励的索引id为句长-1，否则为第一个pad位置-1
            idx = idxs[0].item() + prompt_length if len(idxs) > 0 else seq_len
            v = torch.nn.functional.sigmoid(value[idx - 1])
            scores.append(v)
        return torch.stack(scores)
