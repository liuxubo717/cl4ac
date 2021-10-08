import math
import warnings

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from transformers import AutoModel, AutoTokenizer

from bert_tools.custom_tokenizer import CUSTOM_TOKENIZER
from config_loader.bert_config import BERT_MODELS
from model.Encoder import Cnn10
from w2v_tools.w2v_model import Word2Vec


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        bert_config = config.bert
        self.config = config
        self.bert_config = bert_config
        if self.config.w2v.enable:
            warnings.warn("Word2Vec enabled! BERT relevant config will be useless!")
            self.w2v_model = Word2Vec(w2v_model_path=self.config.w2v.w2v_path, multisos=config.multisos.enable)
            self.bert_config = None
            self.ntoken = len(self.w2v_model.vocabulary)
        dropout = config.decoder.dropout

        if config.encoder.name == 'cnn10':
            self.feature_extractor = Cnn10(config)
        else:
            raise NameError('No such Cnn.')

        # Loading pretrained feature extractor parameters
        pretrained_cnn = torch.load('data/pretrained_models/encoder/' + config.encoder.name + '.pth')['model']

        dict_trained = pretrained_cnn
        dict_new = self.feature_extractor.state_dict().copy()
        if config.encoder.pretrain_loading == 'v2':
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
        else:
            trained_list = [i for i in pretrained_cnn.keys() if not 'fc' in i]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = dict_trained[trained_list[i]]
        self.feature_extractor.load_state_dict(dict_new)
        if config.encoder.freeze_cnn:
            for name, p in self.feature_extractor.named_parameters():
                if not 'fc' in name:
                    p.requires_grad = False

        # Transformer Encoder and Decoder
        if self.bert_config:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_config.bert_path,
                                                           hidden_dropout_prob=dropout,
                                                           output_hidden_states=True)
            if self.bert_config.use_custom_tokenizer:
                self.tokenizer = CUSTOM_TOKENIZER
            self.ntoken = self.tokenizer.get_vocab().__len__()
            # warnings.warn("The ntoken is override by BERT. The new ntoken will be {}".format(self.ntoken))
        self.dropout = nn.Dropout(p=dropout)
        self.nhead = config.decoder.nhead
        self.nhid = config.decoder.nhid

        self.nlayers = config.decoder.nlayers
        self.pos_encoder = PositionalEncoding(self.nhid, dropout)
        self.decoder_only = True

        if not self.decoder_only:
            # encoder
            encoder_layers = TransformerEncoderLayer(d_model=self.nhid, nhead=self.nhead, dim_feedforward=2048,
                                                     dropout=dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model=self.nhid, nhead=self.nhead, dim_feedforward=2048,
                                                dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, self.nlayers)
        self.dec_fc = nn.Linear(self.nhid, self.ntoken)
        self.generator = nn.Softmax(dim=-1)

        if self.bert_config:
            self.bert_model = AutoModel.from_pretrained(bert_config['bert_path'])
            self.word_emb = torch.nn.Linear(BERT_MODELS[bert_config['bert_path']][2], self.nhid)
            if bert_config.use_custom_tokenizer:
                self.bert_model.embeddings.word_embeddings = torch.nn.Embedding(
                    self.ntoken,
                    BERT_MODELS[bert_config['bert_path']][2]
                )
            if config.bert.freeze_bert:
                for name, p in self.bert_model.named_parameters():
                    p.requires_grad = False
        elif self.config.w2v.enable:
            self.word_emb = self.w2v_model.generate_embedding_layer(self.nhid, config.w2v.trainable)
            if self.config.w2v.random_init:
                initrange = 0.1
                self.word_emb.weight.data.uniform_(-initrange, initrange)
        else:
            raise ValueError("Unimplemented Error")

        self.rs_fc = nn.Linear(self.nhid, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.dec_fc.bias.data.zero_()
        self.dec_fc.weight.data.uniform_(-initrange, initrange)

    def encode(self, src):
        src = self.feature_extractor(src)
        # x = src
        if not self.decoder_only:
            src = src * math.sqrt(self.nhid)
            src = self.pos_encoder(src)
            src = self.transformer_encoder(src, None)
        # src = src + x
        return src

    def decode(self, mem, tgt, input_mask=None, target_mask=None, target_padding_mask=None, attention_mask=None,
               selection_result=False, max_non_pad_indexes=None):
        # tgt:(batch_size, T_out)
        # mem:(T_mem, batch_size, nhid)
        if self.config.w2v.enable:
            tgt = tgt.transpose(0, 1)
            if target_mask is None or target_mask.size(0) != len(tgt):
                device = tgt.device
                target_mask = generate_square_subsequent_mask(len(tgt)).to(device)
            tgt = self.word_emb(tgt) * math.sqrt(self.nhid)
            tgt = self.pos_encoder(tgt)
        elif self.config.bert.use_token_level_embedding:
            bert_token_embedding = torch.zeros(
                (tgt.shape[0], tgt.shape[1], BERT_MODELS[self.bert_config['bert_path']][2]))
            bert_token_embedding = bert_token_embedding.to(tgt.device)
            for index in range(tgt.shape[1]):
                self.bert_model.eval()
                with torch.no_grad():
                    current_tgt = tgt[:, 0].unsqueeze(-1)
                    cls_tokens = torch.full(current_tgt.shape, self.tokenizer.cls_token_id).to(current_tgt.device)
                    sep_tokens = torch.full(current_tgt.shape, self.tokenizer.sep_token_id).to(current_tgt.device)
                    current_cls_tgt_sep = torch.cat((cls_tokens, current_tgt, sep_tokens), dim=-1)
                    bert_output = self.bert_model(current_cls_tgt_sep)
                    bert_output = bert_output[0][:, 1, :]
                bert_token_embedding[:, index, :] = bert_output
            embedding_output = self.word_emb(bert_token_embedding)
            tgt = embedding_output.transpose(0, 1)
        else:
            bert_output = self.bert_model(tgt, attention_mask=attention_mask)
            embedding_output = self.word_emb(bert_output[0])
            # transpose before feed into following layers
            tgt = embedding_output.transpose(0, 1)
        if self.config.bert.input_id_all_not_empty and self.bert_config is not None:
            target_mask = generate_square_subsequent_mask(len(tgt)).to(mem.device)
        trans_dec_output = self.transformer_decoder(tgt, mem, memory_mask=input_mask, tgt_mask=target_mask,
                                                    tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(trans_dec_output)
        output = output.transpose(0, 1)
        # this is only for RS calc
        if selection_result:
            if max_non_pad_indexes is not None:
                last_dim = trans_dec_output.shape[-1]
                gather_indexes = max_non_pad_indexes.unsqueeze(1).repeat(1, last_dim).view(-1, 1, last_dim)
                pooling = trans_dec_output.transpose(0,1).gather(1,gather_indexes.type(torch.int64))
                pooling = pooling.view(-1, pooling.shape[-1])
            elif self.config.auxiliary_task.pooling_type == 'max':
                pooling = torch.max(trans_dec_output.transpose(0, 1), 1)[0]
            elif self.config.auxiliary_task.pooling_type == 'mean':
                pooling = torch.mean(trans_dec_output.transpose(0, 1), 1)
            else:
                raise ValueError("Please specify a pooling type")
            rs_score = self.rs_fc(pooling)
            return output, rs_score
        return output

    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None, attention_mask=None,
                selection_result=False, max_non_pad_indexes=None):
        mem = self.encode(src)
        output = self.decode(mem, tgt, input_mask=input_mask, target_mask=target_mask,
                             target_padding_mask=target_padding_mask, attention_mask=attention_mask,
                             selection_result=selection_result,
                             max_non_pad_indexes=max_non_pad_indexes)
        return output
