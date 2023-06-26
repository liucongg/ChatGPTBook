# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
from collections import namedtuple
import random
import os
import csv
import torch
import itertools
import nltk
from nltk import tokenize as nltk_tokenize
from .wordpiece import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


def make_tokenizer(tokenizer_type, corpus, model_path=None, vocab_size=None, model_type=None, pad_token=0,
                   character_coverage=1.0, command_tokens=None, type_tokens=None, fix_command_token=False, **kwargs):
    """
    Helper function to instantiate a tokenizer given common combinations of options.
    """
    return BertWordPieceTokenizer(model_type, **kwargs)


class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenizations without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """

    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def insert(self, idx, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization) - 1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], (CommandToken, TypeToken)):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))

    def __repr__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


DEFAULT_COMMAND_TOKENS = [
    ('pad', 0),
    ('eos', 1),
    ('bos', 2),
    ('unk', 3),
    ('sep', 4),
    ('L2R', 5),
    ('ENC', 6),
    ('MASK', 7),
]
DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)

"""define some default type tokens for bert training"""

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


def prep_type_tokens(tokenlist, token_format=token_format):
    return [TypeToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


DEFAULT_TYPE_TOKENS = [
    ('function', 0),
    ('command', 1),
    ('str0', 2),
    ('str1', 3),
    ('str2', 4),
    ('embedding0', 5),
    ('embedding1', 6),
    ('embedding2', 7),
    ('arg0', 8),
    ('arg1', 9),
    ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        # set command tokens
        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self._command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        # set type tokens
        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        # parse tokens and vocabs from tokenizer
        self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.tokens)
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def token_types(self):
        """list (or iterable) of all token types for tokenizer"""
        return self._token_types

    @property
    def token_type_vocab(self):
        """dictionary mapping token types to ids for tokenizer"""
        return self._token_type_vocab

    @property
    def command_tokens(self):
        """list (or iterable) of all command tokens for tokenizer"""
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.text_tokenizer.EncodeAsTokens(text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        """convert Id to token accounting for command and type tokens"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token
        return self.text_tokenizer.IdToToken(Id - self.num_command_tokens)

    def TokenToId(self, token, type_token=False):
        """convert token to Id accounting for command and type tokens"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id
        return self.text_tokenizer.TokenToId(token) + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        """
        convert Ids to tokens accounting for command and type tokens, tokens
        are joined and returned as a string.
        """
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        rtn_strs = []
        current_str = []
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)


MAX_SENTENCEPIECE_SENTENCES = 100000000


def get_corpus_freq(dataset, filepath, filetype='tsv'):
    """
    Take corpus, split it into sentences, and extract word frequencies.
    Write frequencies to `filepath` as a tsv. Only write the first
    MAX_SENTENCEPIECE_SENTENCES most common words to the file.
    """
    nltk.download('punkt', download_dir="./nltk")
    if filetype == 'tsv':
        delimiter = '\t'
    else:
        delimiter = ','

    print("compute corpus frequency\n", flush=True)

    total_sentence_count = 0
    maxlen = 0
    freqs = {}
    for entry in dataset:
        if isinstance(entry, dict):
            entry = entry['text']
        lines = entry.strip().split('\n')
        for line in lines:
            sentences = nltk_tokenize.sent_tokenize(line)
            total_sentence_count += len(sentences)
            for sentence in sentences:
                maxlen = max(len(line), maxlen)
                for word in sentence.split():
                    if word not in freqs:
                        freqs[word] = 0
                    freqs[word] += 1

    print("length of freqs before truncating " + str(len(freqs)), flush=True)
    print("file path for freq " + str(filepath), flush=True)

    freqs_sorted = {}
    counter = 0
    for word, count in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
        if counter >= MAX_SENTENCEPIECE_SENTENCES:
            break
        counter += 1
        freqs_sorted[word] = count

    print("length of freqs after trancating " + str(len(freqs_sorted)), flush=True)

    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for k, v in freqs_sorted.items():
            writer.writerow([str(k), str(v)])

    return total_sentence_count, maxlen


class BertWordPieceTokenizer(Tokenizer):
    """
    Loads a pretrained WordPiece tokenizer from `cache_dir` for tokenization
    in BERT training. Default to bert-large-uncased tokenizer.
    """

    def __init__(self, tokenizer_model_type=None, cache_dir=None, add_block_symbols=False, add_sentinel_token=0,
                 add_task_mask=False, add_decoder_mask=False, **kwargs):
        # default to bert-large-uncased tokenizer
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('loading BertWordPieceTokenizer (', tokenizer_model_type, ') from cache_dir ', cache_dir)
        do_lower_case = not ('-cased' in tokenizer_model_type or 'chinese' in tokenizer_model_type)
        print("tokenizer_model_type", tokenizer_model_type)
        self.text_tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type, do_lower_case=do_lower_case,
                                                            cache_dir=cache_dir)
        print(self.text_tokenizer.tokenize("我爱北京天安门"))
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('loaded', tokenizer_model_type)
        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)

        # set command tokens from wordpiece tokenizer values
        self.num_command_tokens = 6
        self.num_tokens = len(self.text_tokenizer.vocab)
        self.num_text_tokens = self.num_tokens - 5
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
            CommandToken('ENC', '[CLS]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('MASK', '[MASK]', self.text_tokenizer.vocab['[MASK]']),
            CommandToken('unk', '[UNK]', self.text_tokenizer.vocab['[UNK]']),
            CommandToken('sep', '[SEP]', self.text_tokenizer.vocab['[SEP]']),
            CommandToken('eos', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
        ]
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
            ])
            self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        if add_sentinel_token > 0:
            for i in range(1, add_sentinel_token):
                self._command_tokens.extend([CommandToken(f'MASK{i}', f'[MASK{i}]', self.num_tokens),
                                             CommandToken(f'sop{i}', f'<|startofpiece{i}|>', self.num_tokens + 1)])
                self.num_tokens += 2
                self.num_command_tokens += 2
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        # set type tokens
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # parse tokens and vocabs from tokenizer

        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def EncodeAsTokens(self, text, process_fn=None):
        """convert wordpiece token to Id"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        """convert Id to sentencpiece token"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        """convert sentencpiece token to Id"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        token = token.strip()
        return self.text_tokenizer.vocab[token]

    def DecodeIds(self, Ids, type_token=False):
        """converts ids to wordpiece tokens and joins them as a text string"""
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        Tokens = []
        for Id in Ids:
            if Id in self.command_id_map:
                Tokens.append(self.command_id_map[Id].token)
            elif Id in self.text_tokenizer.ids_to_tokens:
                Tokens.append(self.text_tokenizer.ids_to_tokens[Id])
        new_tokens = []
        for token in Tokens:
            if token.startswith('##') and len(new_tokens) > 0:
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        return ' '.join(new_tokens)

    def DecodeTokens(self, Tokens, type_token=False):
        """converts wordpiece tokens to a text string"""
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ' '.join(Tokens)
