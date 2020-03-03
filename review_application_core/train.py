import os
import numpy as np

from common.data_utils import load_spark, load_data, CONFIG_PATH, CHECKPOINT_PATH, convert_bert_data, TRUE_SAMPLING
from common.tokenizer import BERT_Tokenizer
from common.models import get_bert_model
from common.metrics import All_Metrics
from common.file_utils import write_json
import common.config as config

if __name__ == '__main__':

    args = config.get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    NOW_DIR = os.getcwd()
    if args.model_name == 'bert_siamese':
        DATA_DR = os.path.join(NOW_DIR, 'data', 'False_Exaggerated_advertisement.csv')
    else:
        DATA_DR = os.path.join(NOW_DIR, 'data', 'reddit', 'train-balanced-sarcasm.csv')

    sqlContext = load_spark("local", "{}".format(args.model_name))
    df = load_data(sqlContext, DATA_DR)
    input_cols = ["text", 'categories'] if args.model_name == 'bert_siamese' else ["comment"]
    df = df.na.drop()

    pipeline_model = BERT_Tokenizer(inputCols=input_cols, outputCols=["features"], maxlen=args.seq_len,
                                    model_name=args.model_name)

    df_transform = pipeline_model.transform(df)

    train_test_ratio = [.8, .2] if args.model_name == 'bert_siamese' else [.33, .67]

    train_data, test_data = df_transform.randomSplit(train_test_ratio, seed=1234)

    model = get_bert_model(CONFIG_PATH, CHECKPOINT_PATH, seq_len=args.seq_len, lr=args.lr, model_name=args.model_name)

    train_indices, train_segments, y_train = convert_bert_data(train_data)
    test_indices, test_segments, y_test = convert_bert_data(test_data)

    if args.model_name == 'bert_siamese':
        train_cut_off_value = TRUE_SAMPLING
        test_cut_off_value = [cut_v for cut_v in np.arange(0.1, TRUE_SAMPLING, 0.1)]
    else:
        train_cut_off_value = None
        test_cut_off_value = None

    callback_metrics = All_Metrics([[train_indices, train_segments], y_train], train_cut_off_value, test_cut_off_value)
    model.fit([train_indices, train_segments], y_train, validation_data=([[test_indices, test_segments], y_test]),
              epochs=args.epoch, callbacks=[callback_metrics])

    SAVE_FOLDER = 'FEAD' if args.model_name == 'bert_siamese' else 'SRD'
    SAVE_JSON_DIR = os.path.join(NOW_DIR, 'model', SAVE_FOLDER, args.model_name, 'config.json')
    SAVE_MODEL_DIR = os.path.join(NOW_DIR, 'model', SAVE_FOLDER, args.model_name, 'weight.h5')
    write_json(SAVE_JSON_DIR, args)
    model.save(SAVE_MODEL_DIR)
