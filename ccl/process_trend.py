import sys
import numpy
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
from sklearn import metrics
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.layers.core import Flatten
import random

def load_data(filename, interval):
    try:
        data_info = numpy.loadtxt(filename)
        test_index = int(data_info.shape[0] * 0.8)
        x_data = data_info[0:test_index, 0:interval]
        y_data = data_info[0:test_index, interval:interval+4]
        x_test = data_info[test_index:, 0:interval]
        y_test = data_info[test_index:, interval:interval+4]
        return x_data.reshape(x_data.shape[0], x_data.shape[1], 1), y_data, \
        x_test.reshape(x_test.shape[0],x_data.shape[1], 1),  y_test
    except Exception as e:
        print("info = %s" %(e))
        sys.exit(1)
def build_model(x_data, y_data, x_test, y_test):
    model = Sequential()
    model.add(LSTM(input_shape=[x_data.shape[1], 1],  output_dim =10, return_sequences = True))
    model.add(Flatten())
    model.add(Activation('linear'))
    model.add(Dense(output_dim=20))
    model.add(Activation('linear'))
    model.add(Dense(output_dim=10))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim=4))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_data, y_data, batch_size=1000, nb_epoch=50, validation_split=0.2)
    preds = model.predict_proba(x_data)
    result = metrics.roc_auc_score(y_data, preds)
    print(result)

    preds = model.predict_proba(x_test)
    result = metrics.roc_auc_score(y_test, preds)
    print(result)
    return model
def save_model(model):
    json_string = model.to_json()
    open('trend_model.json','w').write(json_string)    
    model.save_weights('trned_model_weights.h5') 
if __name__ == "__main__":
    x_data, y_data, x_test, y_test = load_data("trend_sample", 15)
    model = build_model(x_data, y_data, x_test, y_test)
    save_model(model)
