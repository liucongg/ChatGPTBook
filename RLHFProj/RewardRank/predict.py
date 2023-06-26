# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/13 13:08
"""
    文件说明：
            
"""
import torch
import argparse
import os

from model import RewardModel
from transformers import BertTokenizer


def predict_one_sample(model, tokenizer, device, args, content, query):
    """
    对单个样本进行预测
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备信息
        args: 配置项信息
        content: 正文
        query: 问题
    Returns:
    """
    # 对文档和问题进行分词处理，并按照最大长度进行裁剪
    content_tokens = tokenizer.tokenize(content)
    query_tokens = tokenizer.tokenize(query)
    if len(query_tokens) > args.max_query_len:
        query_tokens = query_tokens[:args.max_query_len]
    max_content_len = args.max_len - len(query_tokens) - 3
    if len(content_tokens) > max_content_len:
        content_tokens = max_content_len[:max_content_len]
    # 将分词结果转换成模型预测所需的索引内容
    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(content_tokens) + [
        tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(query_tokens) + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    # 进行结果预测
    value = model.predict(input_ids=torch.tensor([input_ids]).to(device),
                          attention_mask=torch.tensor([attention_mask]).to(device),
                          prompt_length=len(content_tokens) + 2)
    score = float(value[0].cpu().detach().numpy())
    return score


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡')
    parser.add_argument('--max_len', default=768, type=int, help='模型最大长度')
    parser.add_argument('--max_query_len', default=64, type=int, help='生成问题最大长度')
    parser.add_argument('--model_path', default="rm_model/", type=str, help='预测模型路径')
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
    model = RewardModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print('开始对文本和问题进行打分，输入CTRL + C，则退出')
    while True:
        content = input("输入的文本为：")
        qeury = input("输入的问题为：")
        score = predict_one_sample(model, tokenizer, device, args, content, qeury)
        print("上述文本和问题的匹配度分数为：{}".format(score))

    # content = "全镇总面积72平方公里，人口1.82万人口。。罗店镇辖境于1959年始设双龙人民公社罗店管理区，1979年改为罗店人民公社。1983年撤消人民公社，改设罗店乡。1986年改设罗店镇。1992年，原双龙乡并入，仍称罗店镇。该镇辖有： 罗店村、前店村、双林村、大舍塘村、林田村、罗大门村、三甲山村、西吴村、后溪河村、上张家村、溪塍村、后园村、童仙村、麻车村、石墙脚村、前庄头村、推包井村、六头塔村、倪西店村、十里铺村、大岭村、羊甲山村、白望山村、西旺村、梅村、建新村、九龙村、玲珑岩村、盘前村、弹子下村、里宅村、洞前村、鹿田村、放生塘村、山下曹村、鹿村、对岳村、长岭村、"
    # query = "全镇总面积多少？"
    # query = "今年CBA总冠军是哪个队"
    # query = "全镇"


if __name__ == '__main__':
    main()
