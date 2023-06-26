# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/14 16:00
"""
    文件说明：
            
"""
from utils import log_probs_from_logits
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel
import torch


class ActorModel(torch.nn.Module):
    """Actor模型"""

    def __init__(self, model_path):
        """初始化模型"""
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def save_pretrained(self, output_dir):
        """模型保存"""
        self.model.save_pretrained(output_dir)

    @torch.no_grad()
    def generate(self, input_ids, **gen_kwargs):
        """
        根据prompt内容生成answer内容
        Args:
            input_ids:
            **gen_kwargs:

        Returns:

        """
        # 根据prompt内容生成结果
        outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)
        # 根据pad_token生成attention_mask，pad部分为0，其他部分为1
        pad_token_id = gen_kwargs.get('pad_token_id', None)
        attention_mask = outputs.not_equal(pad_token_id).to(dtype=torch.long, device=outputs.device)
        return outputs, attention_mask

    def forward(self, input_ids, attention_mask):
        """前馈函数"""
        # 通过模型获取logits输出结果
        logits = self.model.forward(input_ids, attention_mask=attention_mask)['logits']
        # 根据输入获取log_probs概率
        log_probs = log_probs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        return log_probs


class RewardModel(GPT2PreTrainedModel):
    """Reward模型"""

    def __init__(self, config):
        """初始化函数"""
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.value_fn = torch.nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, input_ids, attention_mask, prompt_length):
        """前馈网络"""
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
            # v = torch.sigmoid(value[idx - 1]) * 4
            # scores.append(v)
            scores.append(value[idx - 1])
        scores = torch.stack(scores)
        return values, scores


# Critic模型由Reward模型初始化，因此模型结构一致
CriticModel = RewardModel
