from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


class All_Metrics(Callback):
    def __init__(self, train_data, train_cut_off_value=None, test_cut_off_value=None):
        self.train_cut_off_value = train_cut_off_value
        self.test_cut_off_value = test_cut_off_value
        self.train_data = train_data
        if self.test_cut_off_value:
            self.target_names = ['Fake', 'Real']
        else:
            self.target_names = ['Not_Sarcasm', 'Sarcasm']

    def exchange_y_data(self, y_data, cut_off_value):
        y_data[y_data >= cut_off_value] = 1
        y_data[y_data < cut_off_value] = 0
        return y_data

    def predictions(self, x_data, cut_off_value=None):
        predictions = self.model.predict(x_data)
        if self.target_names[0] == 'Fake':
            predictions[predictions >= cut_off_value] = 1
            predictions[predictions < cut_off_value] = 0
        else:
            predictions = np.argmax(predictions, axis=1)
        return predictions

    def get_classification_report(self, data_type, true, predict, epoch=None, cut_v=None):
        if self.target_names[0] == 'Fake':
            print(
                '------------------------------------------{}-Epoch: {} : {}---------------------'
                '---------------------'.format(
                    data_type, epoch, cut_v))
            print('One_predict : {}'.format(np.count_nonzero(true) / true.shape[0]))
            print('Accuracy : {}'.format(accuracy_score(true, predict)))
            print(classification_report(true, predict, target_names=self.target_names))
        else:
            print(
                '------------------------------------------{}-Epoch: {}--------------------------'
                '----------------'.format(data_type, epoch))
            print('One_predict : {}'.format(1.0 - (np.count_nonzero(true) / true.shape[0])))
            print('Accuracy : {}'.format(accuracy_score(true, predict)))
            print(classification_report(true, predict, target_names=self.target_names))

    def on_epoch_end(self, epoch, logs={}):
        if self.target_names[0] == 'Fake':
            for cut_v in self.test_cut_off_value:
                train_y = self.exchange_y_data(self.train_data[1], self.train_cut_off_value)
                test_y = self.exchange_y_data(self.validation_data[2], self.train_cut_off_value)
                train_predictions = self.predictions(self.train_data[0], cut_v)
                test_predictions = self.predictions([self.validation_data[0], self.validation_data[1]], cut_v)
                self.get_classification_report('train', train_y, train_predictions, epoch, cut_v)
                self.get_classification_report('test', test_y, test_predictions, epoch, cut_v)
            return
        else:
            test_predictions = self.predictions([self.validation_data[0], self.validation_data[1]])
            self.get_classification_report('test', self.validation_data[2], test_predictions, epoch)
            return
