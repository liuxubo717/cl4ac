import torch
from gensim.models import Word2Vec as W2V

from utils.kwargs_helper import get_kwargs_value
from w2v_tools.create_dataset import clean_sentence


class Word2Vec:
    def __init__(self, w2v_model_path='data/pretrained_models/word2vec/w2v.model', multisos=False):
        w2v_model = W2V.load(w2v_model_path)
        self.w2v_model = w2v_model
        self.vocabulary = list(w2v_model.wv.vocab.keys())
        self.vocab_size = len(self.vocabulary)
        # do the similar thing like BERT tokenizer
        self.mask_token = '<eos>'
        self.mask_token_id = self.vocabulary.index(self.mask_token)
        self.pad_token = '<eos>'
        self.pad_token_id = self.vocabulary.index(self.pad_token)
        self.sep_token = '<eos>'
        self.sep_token_id = self.vocabulary.index(self.sep_token)
        if multisos:
            self.cls_tokens = ['<sos0>', '<sos1>', '<sos2>', '<sos3>', '<sos4>']
            self.cls_token_ids = [self.vocabulary.index(cls_t) for cls_t in self.cls_tokens]
        else:
            self.cls_token = '<sos>'
            self.cls_token_id = self.vocabulary.index(self.cls_token)

    def generate_embedding_layer(self, hidden_size, trainable=False):
        weights = torch.randn(len(self.vocabulary), hidden_size)
        for i, word in enumerate(self.vocabulary):
            embedding = self.w2v_model[word]
            weights[i] = torch.from_numpy(embedding)
        word_emb = torch.nn.Embedding.from_pretrained(weights)
        word_emb.weight.requires_grad = trainable
        return word_emb

    def encode_sentence(self, sentence, padding=-1, return_index=True, add_special_tokens=True, caption_index=None):
        cleaned_sentence = clean_sentence(sentence)
        if add_special_tokens:
            if caption_index is not None:
                cleaned_sentence = '<sos{}> {} <eos>'.format(caption_index, cleaned_sentence)
            else:
                cleaned_sentence = '<sos> {} <eos>'.format(cleaned_sentence)
        words = cleaned_sentence.split()
        padding_token = '<eos>'
        if padding > 0 and len(words) < padding:
            words += [padding_token] * (padding - len(words))
        if len(words) > padding > 0:
            raise ValueError("Sentence Length exceeds padding length!")
        indexes = [self.vocabulary.index(w) for w in words]
        if return_index:
            return indexes
        else:
            raise ValueError("Not Implemented Yet!")

    def __call__(self, *args, **kwargs):
        padding = get_kwargs_value(kwargs, 'padding')
        max_length = get_kwargs_value(kwargs, 'max_length')
        return_tensors = get_kwargs_value(kwargs, 'return_tensors')
        add_special_tokens = get_kwargs_value(kwargs, 'add_special_tokens', True)
        caption_index = get_kwargs_value(kwargs, 'caption_index', None)
        assert padding == 'max_length' or padding is False or padding is None, 'Only support max_length padding!'
        if not padding:
            max_length = -1
        text = args[0]
        tokenized = {}
        if text.__class__ == str:
            tokenized['input_ids'] = self.encode_sentence(text, padding=max_length,
                                                          add_special_tokens=add_special_tokens,
                                                          caption_index=caption_index)
        elif text.__class__ == list and len(text) > 0 and text[0].__class__ == str:
            if caption_index is not None:
                tokenized['input_ids'] = [self.encode_sentence(t, padding=max_length,
                                                               add_special_tokens=add_special_tokens,
                                                               caption_index=cap_i) for cap_i, t in
                                          zip(caption_index, text)]
            else:
                tokenized['input_ids'] = [self.encode_sentence(t, padding=max_length,
                                                               add_special_tokens=add_special_tokens)
                                          for t in text]
        else:
            raise ValueError("Not implemented!")
        if return_tensors == 'pt':
            tokenized['input_ids'] = torch.tensor(tokenized['input_ids'])
        return tokenized

    def decode(self, token_ids, skip_special_tokens=False):
        words = [self.vocabulary[index] for index in token_ids]
        if skip_special_tokens:
            special_tokens = ['<eos>', '<sos>', '<sos0>', '<sos1>', '<sos2>', '<sos3>', '<sos4>']
            words = list(filter(lambda x: x not in special_tokens, words))
        return ' '.join(words)

    def batch_decode(self, token_ids, skip_special_tokens=False):
        return [self.decode(tid, skip_special_tokens=skip_special_tokens) for tid in token_ids]
