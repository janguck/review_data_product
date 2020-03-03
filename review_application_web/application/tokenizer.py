import codecs
import os
from keras_bert import Tokenizer

NOW_DIR = os.getcwd()
VOCAB_DR = os.path.join(NOW_DIR, '..', 'review_application_core', 'pretrain', 'bert_base_uncased', 'vocab.txt')


def bert_vocab(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def bert_tokenizer(vocab_path):
    token_dict = bert_vocab(vocab_path)
    return Tokenizer(token_dict)
