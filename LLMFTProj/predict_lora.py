# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: predict_lora
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/5 11:12
"""
    文件说明：
            
"""
import torch
import json
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from peft import PeftModel
from tqdm import tqdm
import time
import os
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/spo_1.json', type=str, help='测试数据')
    parser.add_argument('--device', default='0', type=str, help='模型推理的设备信息')
    parser.add_argument('--ori_model_dir', default="ChatGLM-6B/", type=str, help='预训练模型')
    parser.add_argument('--model_dir', default="output_dir_lora/global_step-2160/", type=str, help='训练后的模型文件')
    parser.add_argument('--max_len', type=int, default=768, help='模型最大输入长度')
    parser.add_argument('--max_src_len', type=int, default=450, help='源文件最大长度')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='提示模板')
    parser.add_argument('--top_p', type=float, default=0.7, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--do_sample', type=bool, default=False, help='是否解码随机采样')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='返回结果个数')
    parser.add_argument('--predict_one', type=bool, default=False, help='返回结果个数')
    return parser.parse_args()


def predict_one_sample(model, tokenizer, args, text):
    # 获取目标文本的最大长度
    max_tgt_len = args.max_len - args.max_src_len - 3
    with torch.no_grad():
        # 对每个样本进行处理，获取模型输入
        src_tokens = tokenizer.tokenize(text)
        prompt_tokens = tokenizer.tokenize(args.prompt_text)
        if len(src_tokens) > args.max_src_len - len(prompt_tokens):
            src_tokens = src_tokens[:args.max_src_len - len(prompt_tokens)]
        tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 将input_ids放在对应设备上
        input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
        # 利用generate函数生成结果内容
        generation_kwargs = {
            "min_length": 5,
            "max_new_tokens": max_tgt_len,
            "top_p": args.top_p,
            "temperature": 0.95,
            "do_sample": args.do_sample,
            "num_return_sequences": args.num_return_sequences,
        }
        response = model.generate_one(input_ids, **generation_kwargs)
        # 进行结果过滤，获取目标文本内容
        res = []
        for i_r in range(generation_kwargs["num_return_sequences"]):
            outputs = response.tolist()[i_r][input_ids.shape[1]:]
            r = tokenizer.decode(outputs).replace("<eop>", "")
            res.append(r)
        # 对结果按照“\n”进行分割，获取每个三元组内容
        pre_res = list(set([rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]))
    return pre_res


def main():
    # 设置预测的配置参数
    args = set_args()
    # 实例化model，并加载Lora模型
    model = ChatGLMForConditionalGeneration.from_pretrained(args.ori_model_dir)
    model = PeftModel.from_pretrained(model, args.model_dir, torch_dtype=torch.float32)
    model.half().to("cuda:{}".format(args.device))
    # 实例化tokenizer
    tokenizer = ChatGLMTokenizer.from_pretrained(args.ori_model_dir)
    # 判断为单条预测还是批量预测
    if args.predict_one:
        # 如果为单条预测，则在命令框中接受文本内容，并将结果打印
        print('开始对文本进行信息抽取，输入CTRL + C，则退出')
        while True:
            text = input("待抽取文本：")
            pre_res = predict_one_sample(model, tokenizer, args, text)
            print("抽取三元组内容为：{}".format(pre_res))
    else:
        # 如果为批量预测，则遍历文件中所有内容，进行逐条预测，计算F1值及总耗时
        save_data = []
        f1, total = 0.0, 0.0
        s_time = time.time()
        # 开始批量推理
        with open(args.test_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(tqdm(fh, desc="iter")):
                total += 1
                # 对每个样本进行处理，获取模型输入
                sample = json.loads(line.strip())
                pre_res = predict_one_sample(model, tokenizer, args, sample["text"])
                real_res = sample["answer"].split("\n")
                # 计算标准结果与预测结果之间的F1值
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                # 将结果添加到save_data中
                save_data.append(
                    {"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": res[0], "f1": f})
        # 计算总预测耗时，以及F1值
        e_time = time.time()
        print("总耗时：{}s".format(e_time - s_time))
        print(f1 / total)
        # 将结果进行保存成json文件，用于后面结果查看
        save_path = os.path.join(args.model_dir, "ft_answer.json")
        fin = open(save_path, "w", encoding="utf-8")
        json.dump(save_data, fin, ensure_ascii=False, indent=4)
        fin.close()


if __name__ == '__main__':
    main()
