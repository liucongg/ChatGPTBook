# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: chatbot
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:42
"""
    文件说明：
            
"""
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from modeling_unilm import UnilmForSeq2SeqDecodeSample, UnilmConfig
import copy
import os
import argparse
import re
from dirty_recognize import dirty_reg


def remove_dirty_sentence(dirty_obj, sentence):
    if len(dirty_obj.match(sentence)) == 0:
        return False
    else:
        return True


def remove_multi_symbol(text):
    r = re.compile(r'([.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-])[.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-]+')
    text = r.sub(r'\1', text)
    return text


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str, help='生成设备')
    parser.add_argument('--topk', default=3, type=int, help='取前k个词')
    parser.add_argument('--topp', default=0.95, type=float, help='取超过p的词')
    parser.add_argument('--dirty_path', default='data/dirty_words.txt', type=str, help='敏感词库')
    parser.add_argument('--model_name_or_path', default='kuakua_robot_model/', type=str, help='模型路径')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help="重复词的惩罚项")
    parser.add_argument('--max_len', type=int, default=32, help='生成的对话的最大长度')

    args = parser.parse_args()

    # 加载已训练完成模型，并将其放置到对应的设备上
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(device)

    config = UnilmConfig.from_pretrained(args.model_name_or_path, max_position_embeddings=512)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = UnilmForSeq2SeqDecodeSample.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    model.eval()
    print('Chitchat Robot Starting')
    # 加载敏感词过滤器
    dirty_obj = dirty_reg(args.dirty_path)
    while True:
        # 接收用户输入，如果用户输入中包含敏感词的话，直接回复固定话术
        text = input("user:")
        if remove_dirty_sentence(dirty_obj, text):
            print("chat-bot:" + "换个话题聊聊吧。")
            continue
        # 否则，对用户输入进行tokenizer处理，获取对应的token_type
        input_ids = tokenizer.encode(text)
        token_type_ids = [4] * len(input_ids)
        generated = []
        # 针对最大输出长度进行遍历
        for _ in range(args.max_len):
            # 获取当前输入的输出向量
            curr_input_ids = copy.deepcopy(input_ids)
            curr_input_ids.append(tokenizer.mask_token_id)
            curr_input_tensor = torch.tensor(curr_input_ids).long().to(device).view([1, -1])
            curr_token_type_ids = copy.deepcopy(token_type_ids)
            curr_token_type_ids.extend([5])
            curr_token_type_ids = torch.tensor(curr_token_type_ids).long().to(device).view([1, -1])
            outputs = model(input_ids=curr_input_tensor, token_type_ids=curr_token_type_ids, attention_mask=None)
            next_token_logits = outputs[-1, -1, :]
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            # 采用top_k和top_p解码，选取词表中的Token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 当遇到[SEP]则表明生成结束
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明生成结束
                break
            generated.append(next_token.item())
            input_ids.append(next_token.item())
            token_type_ids.extend([5])
        # 将生成的所有内容进行拼接，并检测是否存在敏感词，如果存在敏感词回复固定话术，否则回复模型生成内容
        text = tokenizer.convert_ids_to_tokens(generated)
        text = remove_multi_symbol("".join(text))
        if remove_dirty_sentence(dirty_obj, text):
            print("chat-bot:" + "我要想一想。")
        else:
            print("chat-bot:" + text)


if __name__ == "__main__":
    main()
