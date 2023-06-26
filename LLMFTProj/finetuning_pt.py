# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: finetuning_pt
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/4 14:40
"""
    文件说明：
            
"""
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from configuration_chatglm import ChatGLMConfig
import torch
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
from data_set import Seq2SeqDataSet, coll_fn, print_trainable_parameters
import os
from shutil import copy


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/spo_0.json', type=str, help='训练文件')
    # parser.add_argument('--model_dir', default="ChatGLM-6B/", type=str, help='预训练模型')
    parser.add_argument('--model_dir', default="/data/work/lcong/public_model_path/ChatGLM-6B/", type=str, help='预训练模型')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练轮数')
    parser.add_argument('--train_batch_size', default=2, type=int, help='训练批次')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度累积步数')
    parser.add_argument('--output_dir', default='output_dir_pt/', type=str, help='模型保存路径')
    parser.add_argument('--log_steps', type=int, default=10, help='日志打印步数')
    parser.add_argument('--max_len', type=int, default=768, help='模型最大长度')
    parser.add_argument('--max_src_len', type=int, default=450, help='源文本最大长度')
    parser.add_argument('--pre_seq_len', type=int, default=16, help='soft-prompt长度')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='是否在模型层添加参数')
    parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed所需进程相对序号')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='提示模板')
    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    # 实例化ChatGLMConfig文件，设置pre_seq_len和prefix_projection参数
    config = ChatGLMConfig.from_pretrained(args.model_dir)
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection

    # 实例化ChatGLM模型，并用半精度加载模型
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir, config=config)
    model = model.half().cuda()
    model.gradient_checkpointing_enable()

    # 设置仅训练prefix_encoder层参数
    for name, param in model.named_parameters():
        if not any(nd in name for nd in ["prefix_encoder"]):
            param.requires_grad = False

    # 打印模型总参数，以及训练参数占比，并打印可以训练参数名称
    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    # 实例化tokenizer
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)

    # 设置DeepSpeed配置参数，并进行DeepSpeed初始化
    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()

    # 加载训练数据
    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)

    # 开始模型训练
    global_step = 0
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            # 获取训练结果
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs[0]
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            # 损失进行回传
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            # 如果步数整除log_steps，则打印损失值，方便观察
            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
        # 每一轮模型训练结束，进行模型保存
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model.save_pretrained(save_dir)
        copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_pt.py
