# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/3/21 23:02
"""
    文件说明：
            
"""
import torch
import os
import argparse
from model import PromptModel
from transformers import BertTokenizer


def get_verbalizer(label_dict, tokenizer):
    """根据答案空间映射字典获取答案字典id和对应mask向量"""
    # 获取标签词
    label_words = []
    for label, verbalizer in label_dict.items():
        label_words.append(verbalizer["label_words"])

    all_ids = []
    # 遍历每个标签的标签词，构建标签词id列表
    for words_per_label in label_words:
        ids_per_label = []
        for word in words_per_label:
            ids = tokenizer.encode(word, add_special_tokens=False)
            ids_per_label.append(ids)
        all_ids.append(ids_per_label)
    # 判断单个标签词最大长度
    max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
    # 判断每个类别最大标签词个数
    max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
    # 获取标签词id列表对应的掩码列表
    words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                      + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                      for ids_per_label in all_ids]
    # 将标签词id列表进行填充
    words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                 + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                 for ids_per_label in all_ids]
    # 返回标签词id及掩码矩阵，用于答案空间映射
    return torch.tensor(words_ids), torch.tensor(words_ids_mask)


def predict_one_sample(args, device, model, tokenizer, template, text, words_ids, words_ids_mask):
    """单条文本预测函数"""
    # 当评论数据过长时，进行切断操作
    if len(text) > args.max_len - len(template):
        text = text[:args.max_len - len(template)]
    # 将prompt模板与评论数据融合
    text = "[CLS]" + template.replace("{mask}", "[MASK]").replace("{text}", text) + "[SEP]"
    # 对数据进行tokenize分词
    text_tokens = tokenizer.tokenize(text)
    # 获取[MASK]位置，用于填词
    mask_index = text_tokens.index("[MASK]")
    # 生成模型训练所需的input_ids和attention_mask
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # 生成推理模型所需输入矩阵
    input_ids = torch.tensor([input_ids]).to(device)
    attention_mask = torch.tensor([attention_mask]).to(device)
    mask_index = torch.tensor([mask_index]).to(device)
    # 获取预测结果
    outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, mask_index=mask_index,
                            token_handler=args.token_handler, words_ids=words_ids, words_ids_mask=words_ids_mask)
    # 获取模型预测结果
    pre_label = outputs[1].cpu().numpy().tolist()[0]
    return pre_label


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--model_path', default='prompt_model/', type=str, help='prompt模型文件路径')
    parser.add_argument('--max_len', type=int, default=256, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--token_handler', type=str, default="mean", help='答案映射标签多token策略')
    parser.add_argument('--template', type=str, default="{mask}满意。{text}", help='prompt模板')
    parser.add_argument('--pos_words', type=list, default=["很", "非常"], help='答案映射正标签对应标签词')
    parser.add_argument('--neg_words', type=list, default=["不"], help='答案映射负标签对应标签词')
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
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    model = PromptModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    # 获取答案空间映射向量
    label_dict = {"负向": {"label_words": args.neg_words, "label_id": 0},
                  "正向": {"label_words": args.pos_words, "label_id": 1}}
    words_ids, words_ids_mask = get_verbalizer(label_dict, tokenizer)
    words_ids, words_ids_mask = words_ids.to(device), words_ids_mask.to(device)
    print('开始对评论数据进行情感分析，输入CTRL + C，则退出')
    while True:
        text = input("输入的评论数据为：")
        # 对每个文本进行预测
        pre_label = predict_one_sample(args, device, model, tokenizer, args.template, text, words_ids, words_ids_mask)
        if pre_label == 0:
            label = "负向"
        else:
            label = "正向"
        print("情感极性为：{}".format(label))


if __name__ == '__main__':
    main()
