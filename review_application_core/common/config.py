import os
import argparse

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, choices='bert_siamese bert_sarcasm'.split(), help='Name of '
                                                                                                         'model Name')
    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--seq_len', default=64, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)

    return parser.parse_args()
