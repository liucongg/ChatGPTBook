# -*- coding: utf-8 -*- 
# @Time : 2023/4/11 11:33 
# @Author : JunkRoy 
# @E-mail: shenroy92@gmail.com
# @File : model.py
from model.modeling_glm import GLMModel


def get_model(args):
    model = GLMModel(num_layers=args.num_layers,
                     vocab_size=args.vocab_size,
                     hidden_size=args.hidden_size,
                     num_attention_heads=args.num_attention_heads,
                     embedding_dropout_prob=args.hidden_dropout,
                     attention_dropout_prob=args.attention_dropout,
                     output_dropout_prob=args.hidden_dropout,
                     max_sequence_length=args.max_position_embeddings,
                     max_memory_length=args.mem_length,
                     checkpoint_activations=args.checkpoint_activations,
                     checkpoint_num_layers=args.checkpoint_num_layers,
                     parallel_output=False,
                     relative_encoding=args.transformer_xl,
                     block_position_encoding=args.block_lm and not args.masked_lm,
                     output_predict=True,
                     spell_length=None,
                     spell_func=args.prompt_func,
                     attention_scale=args.attention_scale)

    if args.freeze_transformer:
        model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
    return model
