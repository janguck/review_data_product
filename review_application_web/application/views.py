import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'review_application_core'))
from .serializers import SRDSerializer, FEADSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from keras.models import load_model
from keras_bert import get_custom_objects
from .tokenizer import bert_tokenizer, VOCAB_DR
from common.file_utils import read_json
import numpy as np
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

NOW_DIR = os.getcwd()
SAVE_JSON_DIR_F = os.path.join(NOW_DIR, '..', 'review_application_core', 'model', 'FEAD', 'bert_siamese', 'config.json')
SAVE_MODEL_DIR_F = os.path.join(NOW_DIR, '..', 'review_application_core', 'model', 'FEAD', 'bert_siamese', 'weight.h5')
model_siamese = load_model(SAVE_MODEL_DIR_F, custom_objects=get_custom_objects())

SAVE_JSON_DIR_S = os.path.join(NOW_DIR, '..', 'review_application_core', 'model', 'SRD', 'bert_sarcasm', 'config.json')
SAVE_MODEL_DIR_S = os.path.join(NOW_DIR, '..', 'review_application_core', 'model', 'SRD', 'bert_sarcasm', 'weight.h5')

model_sarcasm = load_model(SAVE_MODEL_DIR_S, custom_objects=get_custom_objects())

config_f = read_json(SAVE_JSON_DIR_F)
config_s = read_json(SAVE_JSON_DIR_S)

tokenizer = bert_tokenizer(VOCAB_DR)


class SRDViewSet(APIView):
    keys = ['review']
    label_name = ['Not_Sarcasm', 'Sarcasm']
    graph = tf.get_default_graph()

    def get(self, request):

        input_variables = {k: request.GET.get(k) for k in self.keys}
        ids, segments = tokenizer.encode(input_variables['review'], max_len=config_s['seq_len'])
        x_data = [np.array([ids]), np.array([segments])]
        global model_sarcasm
        with self.graph.as_default():
            predictions = model_sarcasm.predict(x_data)
        input_variables['predict_label'] = self.label_name[np.argmax(predictions, axis=1)[0]]
        serializer = SRDSerializer(data=input_variables)
        if serializer.is_valid():
            serializer.save()
        return Response(input_variables)


class FEADViewSet(APIView):
    keys = ['review', 'description']
    graph = tf.get_default_graph()

    def get(self, request):
        input_variables = {k: request.GET.get(k) for k in self.keys}
        ids, segments = tokenizer.encode(first=input_variables['review'],
                                         second=input_variables['description'], max_len=config_f['seq_len'])
        x_data = [np.array([ids]), np.array([segments])]
        global model_siamese
        with self.graph.as_default():
            predictions = model_siamese.predict(x_data)
        predictions_label = 'False_Advertisement' if predictions[0] < 0.4 else 'True_Advertisement'
        input_variables['predict_label'] = predictions_label
        serializer = FEADSerializer(data=input_variables)
        if serializer.is_valid():
            serializer.save()

        return Response(input_variables)
