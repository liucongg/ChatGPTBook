# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/3 14:03
"""
    文件说明：
            
"""
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.models.gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        """
        初始化函数
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, labels=None):
        """
        前向函数，计算GPT2预测结果值
        Args:
            input_ids: 输入序列在词表中的索引序列，size: [batch_size, sequence_length]
            labels: 标签序列，size: [batch_size, sequence_length]，一般情况下，与input_ids相同

        Returns:

        """
        # 获取GPT2模型的输出结果
        transformer_outputs = self.transformer(input_ids)
        # 获取GPT2模型的最后一层的隐层节点状态，size:[batch_size, sequence_length, config.n_embd]
        hidden_states = transformer_outputs[0]
        # 预测隐层节点状态中的每一个token的下一个token，size:[batch_size, sequence_length, config.vocab_size]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,)
        # 如果labels不为None时，计算损失值loss，并拼接到输出结果中
        if labels is not None:
            # 对预测结果和标签进行偏移操作
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()
            # 获取真实长度，去除padding部分，并计算真实loss
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs
