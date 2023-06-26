# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:27
"""
    文件说明：
            
"""
from random import randint, shuffle
from random import random as rand
import math
import torch
import torch.utils.data
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    if len(tokens_a) + len(tokens_b) > max_len - 3:
        while len(tokens_a) + len(tokens_b) > max_len - 3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


def truncate_tokens_signle(tokens_a, max_len):
    if len(tokens_a) > max_len - 2:
        tokens_a = tokens_a[:max_len - 2]
    return tokens_a


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False,
                 bi_uni_pipeline=None):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.ex_list = []
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(p.imap(annotate_, file_data.readlines(), chunksize=32),
                     total=len(file_data.readlines()),
                     desc="convert seq2seq example", )
            )
        print('Load {0} data'.format(len(self.ex_list)))

    def read_data(self, line):
        sample = eval(line.strip())
        src_tk = sample["src_text"]
        tgt_tk = sample["tgt_text"]
        return src_tk, tgt_tk

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance

    def __iter__(self):
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq:
    """模型数据处理类"""

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0,
                 mask_whole_word=False, mask_source_words=True, tokenizer=None):
        self.max_len = max_len
        self.max_pred = max_pred
        self.mask_prob = mask_prob
        self.vocab_words = vocab_words
        self.indexer = indexer
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        # 获取句子A和句子B，并将其进行tokenize操作
        next_sentence_label = None
        tokens_a, tokens_b = instance[:2]
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # 根据给定的最大长度，对其进行截取
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # 通过特殊字符将其进行拼接
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
        # 获取可以被掩码的真实长度，在Seq2Seq阶段，仅对目标句子进行掩码操作，原始句子不进行掩码操作
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(1, int(round(effective_length * self.mask_prob))))
        # 获取所有Token的位置信息，用于后续掩码操作
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            if (i >= len(tokens_a) + 2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a) + 2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        # 选取待掩蔽Token的位置信息
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue
            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end
            # n-gram掩码
            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # 全词掩码
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    # 随机掩码
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break
        # 根据最大掩码个数，筛选出真实掩码位置
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        # 80%采用[MASK]标记替换，10%采用字典中随机Token替换，10%不替换
        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)

        # 将输入进行Tensor转换，包含模型所需的input_ids、segment_ids、input_mask等
        masked_weights = [1] * len(masked_tokens)
        masked_ids = self.indexer(masked_tokens)
        input_ids = self.indexer(tokens)
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a) + 2].fill_(1)
        second_st, second_end = len(
            tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0] * n_pad)
            if masked_pos is not None:
                masked_pos.extend([0] * n_pad)
            if masked_weights is not None:
                masked_weights.extend([0] * n_pad)

        return input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label
