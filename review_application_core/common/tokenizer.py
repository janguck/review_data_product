import codecs
import os
from keras_bert import Tokenizer
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Params, Param
from pyspark import keyword_only
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.sql import functions as py_f

NOW_DIR = os.getcwd()
VOCAB_DR = os.path.join(NOW_DIR, 'pretrain', 'bert_base_uncased', 'vocab.txt')

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


class BERT_Tokenizer(Transformer, HasInputCols, HasOutputCols):

    maxlen = Param(Params._dummy(), "maxlen", "max len to fill")
    model_name = Param(Params._dummy(), "model_name", "model name to fill")

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, maxlen=64, model_name='bert_siamese'):
        super(BERT_Tokenizer, self).__init__()
        self._setDefault(maxlen=64, model_name='bert_siamese')
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, maxlen=64, model_name='bert_siamese'):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setMaxlen(self, maxlen):
        return self._set(maxlen=maxlen)

    def getMaxlen(self):
        return self.getOrDefault(self.maxlen)

    def setModelName(self, model_name):
        return self._set(model_name=model_name)

    def getModelName(self):
        return self.getOrDefault(self.model_name)

    def _transform(self, dataset):
        global tokenizer
        tokenizer = bert_tokenizer(VOCAB_DR)

        def f(s):
            if self.getModelName() == 'bert_siamese':
                ids, segments = tokenizer.encode(first=s[0].lower(), second=s[1].lower(), max_len=self.getMaxlen())
            else:
                ids, segments = tokenizer.encode(s[0].lower(), max_len=self.getMaxlen())
            return [ids, segments]

        in_cols = dataset[self.getInputCols()]
        out_cols = self.getOutputCols()
        convert_udf = py_f.udf(f, ArrayType(ArrayType(IntegerType())))
        convert_s_to_ids = convert_udf(py_f.array([i for i in in_cols]))
        for out_col in out_cols:
            dataset = dataset.withColumn(out_col, convert_s_to_ids)
        if self.getModelName() == 'bert_siamese':
            dataset = dataset.withColumn('label', py_f.regexp_replace('label', 'True_Sampling', '0.6'))
            dataset = dataset.withColumn('label', py_f.regexp_replace('label', 'Soft_Negative_Sampling', '0.2'))
            dataset = dataset.withColumn('label', py_f.regexp_replace('label', 'Hard_Negative_Sampling', '0.1'))
            dataset = dataset.withColumn('label', dataset['label'].cast(FloatType()))
            dataset = dataset.select(dataset.features[0].alias('indices'), dataset.features[1].alias('segments'),
                                     py_f.round(py_f.col('label'), 1).alias('label'))
        else:
            dataset = dataset.select(dataset.features[0].alias('indices'), dataset.features[1].alias('segments'),
                                     py_f.round(py_f.col('label'), 1).alias('label'))

        return dataset
