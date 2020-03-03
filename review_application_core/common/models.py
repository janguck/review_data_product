from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint


MODEL_CONFIG = {'bert_siamese': {'output_dim': 1, 'activation': 'sigmoid', 'loss': 'binary_crossentropy'},
                'bert_sarcasm': {'output_dim': 2, 'activation': 'softmax', 'loss': 'sparse_categorical_crossentropy'}}


def get_bert_model(config_path, checkpoint_path, seq_len, lr, model_name):
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, trainable=True,
                                               seq_len=seq_len)
    model_config = MODEL_CONFIG[model_name]
    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = Dense(units=model_config['output_dim'], activation=model_config['activation'])(dense)
    model = Model(inputs, outputs)
    model.compile(Adam(lr=lr), loss=model_config['loss'])
    return model
