# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: utils
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/14 16:02
"""
    文件说明：
            
"""
import torch


def log_probs_from_logits(logits, labels):
    """
    根据模型预测的logits和标签，获取对应标签位置概率
    Args:
        logits:
        labels:

    Returns:

    """
    probs = torch.log_softmax(logits, dim=-1)
    probs_labels = probs.gather(dim=-1, index=labels.unsqueeze(-1))
    probs_labels = probs_labels.squeeze(-1)
    return probs_labels


def get_advantages_and_returns(values, rewards, start_ids, gamma, lam):
    """
    计算奖励内容的优势函数和返回值
    参考：https://github.com/lvwerra/trl/blob/a004b02c4a/trl/trainer/ppo_trainer.py
    Args:
        values:
        rewards:
        start_ids:
        gamma:
        lam:

    Returns:

    """
    lastgaelam = 0
    advantages_reversed = []
    seq_len = rewards.size()[-1]
    for t in reversed(range(start_ids, seq_len)):
        nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1).detach()
    returns = advantages + values[:, start_ids:]
    return advantages, returns


def actor_loss_function(logprobs, old_logprobs, advantages, actions_mask, policy_clip_eps):
    """
    计算actor模型的损失值
    Args:
        logprobs:
        old_logprobs:
        advantages:
        actions_mask:
        policy_clip_eps:

    Returns:

    """
    log_ratio = (logprobs - old_logprobs) * actions_mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - policy_clip_eps, 1.0 + policy_clip_eps)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * actions_mask) / actions_mask.sum()
    return pg_loss


def critic_loss_function(values, old_values, returns, actions_mask, value_clip_eps):
    """
    计算critic模型的损失值
    Args:
        values:
        old_values:
        returns:
        actions_mask:
        value_clip_eps:

    Returns:

    """
    values_clipped = torch.clamp(values, old_values - value_clip_eps, old_values + value_clip_eps, )
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * actions_mask) / actions_mask.sum()
    return vf_loss
