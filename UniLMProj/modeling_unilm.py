# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: modeling_unilm
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:11
"""
    文件说明：
            
"""
import copy
import math
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from transformers.modeling_utils import PreTrainedModel
from configuration_unilm import UnilmConfig
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, BertPooler, BertIntermediate, BertOutput, \
    BertSelfOutput, BertOnlyMLMHead, BertEmbeddings

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None):
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(
                    hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UnilmPreTrainedModel(PreTrainedModel):
    config_class = UnilmConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "unilm"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class UnilmModel(UnilmPreTrainedModel):
    def __init__(self, config):
        super(UnilmModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class LabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None,
                 reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob.type_as(output), reduction='none').view(batch_size, num_pos, -1).sum(2)


class UnilmForSeq2Seq(UnilmPreTrainedModel):
    """UniLM模型进行Seq2Seq的训练模型类"""

    def __init__(self, config):
        """模型初始化函数，定义模型训练所需的各个模块"""
        super(UnilmForSeq2Seq, self).__init__(config)
        self.bert = UnilmModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.mask_lm = nn.CrossEntropyLoss(reduction='none')
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.mask_lm_smoothed = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=0,
                                                       reduction='none')
        else:
            self.mask_lm_smoothed = None
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """权重加载，加载预训练模型的embeddings部分权重"""
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, masked_pos=None,
                masked_weights=None):
        """模型forward，向前传递函数"""
        # 获取Encoder部分的序列输出，维度[bs,seq_len,hidden_size]
        sequence_output, __ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores = self.cls(sequence_output)
            else:
                sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos)
                prediction_scores = self.cls(sequence_output_masked)
            return prediction_scores
        # 获取被掩码位置的向量
        sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos)
        prediction_scores_masked = self.cls(sequence_output_masked)
        if self.mask_lm_smoothed:
            masked_lm_loss = self.mask_lm_smoothed(F.log_softmax(prediction_scores_masked.float(), dim=-1),
                                                   masked_lm_labels)
        else:
            masked_lm_loss = self.mask_lm(prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        # 计算[Mask]标记的损失值
        masked_lm_loss = loss_mask_and_normalize(masked_lm_loss.float(), masked_weights)

        return masked_lm_loss


class UnilmForSeq2SeqDecodeSample(UnilmPreTrainedModel):
    """UniLM模型进行Seq2Seq的模型解码类"""
    def __init__(self, config):
        """模型初始化函数，定义模型训练所需的各个模块"""
        super(UnilmForSeq2SeqDecodeSample, self).__init__(config)
        self.bert = UnilmModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """权重加载，加载预训练模型的embeddings部分权重"""
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 获取Encoder部分的序列输出，维度[bs,seq_len,hidden_size]
        sequence_output, __ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # 获取最优一个节点的输出
        last_hidden = sequence_output[:, -1:, :]
        # 将其映射到词表中，为后面解码提供内容
        prediction_scores = self.cls(last_hidden)
        return prediction_scores
