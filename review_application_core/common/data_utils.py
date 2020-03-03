import os
import numpy as np
from nltk.tokenize import RegexpTokenizer
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

NOW_DIR = os.getcwd()
VOCAB_DR = os.path.join(NOW_DIR, 'pretrain', 'bert_base_uncased', 'vocab.txt')
CONFIG_PATH = os.path.join(NOW_DIR, 'pretrain', 'bert_base_uncased', 'bert_config.json')
CHECKPOINT_PATH = os.path.join(NOW_DIR, 'pretrain', 'bert_base_uncased', 'bert_model.ckpt')


JACCARD_SCORE = 0.05
TRUE_SAMPLING = 0.6
HARD_NEGATIVE_SAMPLING = 0.1
SOFT_NEGATIVE_SAMPLING = 0.2


def sent_TokenizeFunct(x):
    retokenize = RegexpTokenizer("[\w]+")
    return retokenize.tokenize(x.lower())


def jaccard(x):
    set_text = set(sent_TokenizeFunct(x[0]))
    set_category = set(sent_TokenizeFunct(x[1]))
    intersection_token = set_text.intersection(set_category)
    score = float(len(intersection_token)) / (len(set_text) + len(set_category) - len(intersection_token))
    return score


def data_sampling(x):
    score = jaccard([x.text, x.categories])
    if score > JACCARD_SCORE:
        if x.review_stars>x.business_stars:
            return x.text, x.categories, 'True_Sampling'
        else:
            if x.review_stars < x.average_stars:
                return x.text, x.categories, 'Hard_Negative_Sampling'
            else:
                return x.text, x.categories, 'Soft_Negative_Sampling'
    else:
        return


def load_spark(master, appName):
    sc = SparkContext(master=master, appName=appName)
    return SQLContext(sc)


def load_data(spark, dr):
    return spark.read.csv(dr, header=True, inferSchema=True)


def convert_bert_data(data, purpose='train'):
    data = data.collect()
    data_indices = np.array([_data.indices for _data in data])
    data_segments = np.array([_data.segments for _data in data])
    if purpose == 'train':
        y_data = np.array([_data.label for _data in data])
    else:
        y_data = []
    return data_indices, data_segments, y_data
