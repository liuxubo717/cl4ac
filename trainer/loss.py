import string
import nltk
import torch
from nltk.corpus import stopwords
from torch import nn


def get_loss(config):
    if config.criteria.loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Please use a valid loss")


def calculating_weight(tokenizer, addition_weight_ratio, reduce_punc, reduce_stopwords):
    nltk.download('stopwords')
    stopwords.words('english')
    weight = [1.0] * tokenizer.vocab_size
    # set the word with dataset's distribution
    for token in addition_weight_ratio.keys():
        weight[token] = addition_weight_ratio[token]
    # reduce the weight of [PAD], [SEP]
    # weight[tokenizer.sep_token_id] = 0.1
    weight[tokenizer.pad_token_id] = 0.1
    # reduce the weight of punctuations
    if reduce_punc:
        for punc in string.punctuation:
            tokenized_stop = tokenizer.encode(punc, add_special_tokens=False)
            if len(tokenized_stop) == 1:  # Ignore BPE encoding
                weight[tokenized_stop[0]] = 0.1
    # reduce the weight of common word
    if reduce_stopwords:
        for word in stopwords.words():
            tokenized_stop = tokenizer.encode(word, add_special_tokens=False)
            if len(tokenized_stop) == 1:  # Ignore BPE encoding
                weight[tokenized_stop[0]] = 0.5
    return weight
