# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:39
"""
    文件说明：
            
"""
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from transformers import BertTokenizer
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig
from transformers import AdamW, get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import data_set

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="2", type=str, help="")
    parser.add_argument("--data_dir", default="data/", type=str, help="")
    parser.add_argument("--src_file", default="train.json", type=str, help="")
    parser.add_argument("--model_name_or_path", default="pretrain_model/", type=str, help="")
    parser.add_argument("--output_dir", default="output_dir", type=str, help="")
    parser.add_argument("--max_seq_length", default=256, type=int, help="")
    parser.add_argument("--do_lower_case", default=True, type=bool, help="")
    parser.add_argument("--train_batch_size", default=16, type=int, help="")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="")
    parser.add_argument('--seed', type=int, default=42, help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="")
    parser.add_argument("--mask_prob", default=0.20, type=float, help="")
    parser.add_argument('--max_pred', type=int, default=20, help="")
    parser.add_argument("--num_workers", default=0, type=int, help="")
    parser.add_argument('--mask_source_words', action='store_true', help="")
    parser.add_argument('--skipgram_prb', type=float, default=0.0, help='')
    parser.add_argument('--skipgram_size', type=int, default=1, help='')
    parser.add_argument('--mask_whole_word', type=bool, default=True, help="")
    parser.add_argument('--logging_steps', type=int, default=5, help='')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tb_writer = SummaryWriter()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 模型加载
    config = UnilmConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = UnilmForSeq2Seq.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    # 模型所需数据处理
    print("Loading Train Dataset", args.data_dir)
    bi_uni_pipeline = [data_set.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys()),
                                                   tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                                   mask_source_words=False, skipgram_prb=args.skipgram_prb,
                                                   skipgram_size=args.skipgram_size,
                                                   mask_whole_word=args.mask_whole_word, tokenizer=tokenizer)]

    file = os.path.join(args.data_dir, args.src_file)
    train_dataset = data_set.Seq2SeqDataset(
        file, args.train_batch_size, tokenizer, args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline)

    train_sampler = RandomSampler(train_dataset, replacement=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=args.num_workers,
                                                   collate_fn=data_set.batch_list_to_batch_tensors,
                                                   pin_memory=False)

    # 训练AdamW优化器设置
    t_total = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", t_total)
    # 模型训练
    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # 遍历每个Epoch
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        # 遍历每个Batch数据
        iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)', disable=False)
        for step, batch in enumerate(iter_bar):
            batch = [t.to(device) if t is not None else None for t in batch]
            input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
            # 损失计算
            masked_lm_loss = model(input_ids, segment_ids, input_mask, lm_label_ids,
                                   masked_pos=masked_pos, masked_weights=masked_weights)

            loss = masked_lm_loss
            tr_loss += loss.item()
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 模型参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
        # 每一个Epoch进行模型保存
        logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
