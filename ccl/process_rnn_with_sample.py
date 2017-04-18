import sys
import pandas
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
from keras.models import model_from_json
from keras import initializers
from keras import utils
from keras.layers import advanced_activations 
sample_bdate = ''
sample_edate = ''
test_bdate = ''
test_edate = ''
def weighted_crossentropy(predictions, targets):
    loss = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(predictions, targets))
    return loss
def get_sample(data_info, sample_bdate, sample_edate):
    global interval
    try:
        x_data = data_info[data_info.date>sample_bdate]
        x_data = x_data[x_data.date<sample_edate]
        x_data1 = utils.to_categorical(x_data.iloc[:, 0].values, num_classes = 500)
        x_data2 = x_data.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
        x_data3 = x_data2/x_data2[:, 0].reshape(x_data.shape[0],1)
        x_data4 = x_data3
        y_data = data_info[data_info.date>sample_bdate]
        y_data = y_data[y_data.date<sample_edate].iloc[:, 17].values
        return x_data4, y_data
    except Exception as e:
             print("tmp info = %s" %(e))
             sys.exit(1)
def load_data(filename):
    global sample_bdate
    global sample_edate
    global test_bdate
    global test_edate
    try:
        data_info = pandas.read_csv(filename, sep = ' ', header=0, names=['sym', 'date', 'price15', 'price14', \
                'price13', 'price12', 'price11', 'price10', 'price9', 'price8', 'price7', 'price6', 'price5', 'price4', 'price3','price2', 'price1', 'label'])
        x_data, y_data = get_sample(data_info, sample_bdate, sample_edate)
        x_test, y_test = get_sample(data_info, test_bdate, test_edate)
        return x_data, y_data, x_test, y_test
    except Exception as e:
        print("info = %s" %(e))
        sys.exit(1)



def build_model(x_data, y_data, x_test, y_test):
    print(x_data[0])
    print(y_data[0])
    model = Sequential()
    model.add(Dense(input_shape=[x_data.shape[1]],output_dim=300, kernel_initializer= initializers.Zeros()))
    model.add(Activation('linear'))
#    model.add(Dropout(0.1))
#    model.add(Activation('relu'))
    model.add(Dense(output_dim=100))
    model.add(Activation('tanh'))
#    model.add(Dense(output_dim=100))
#    model.add(Activation('relu'))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.2, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.fit(x_data, y_data, batch_size=10000, nb_epoch=20, shuffle=True, validation_split=0.2)
    preds = model.predict_proba(x_data)
    result = metrics.roc_auc_score(y_data, preds)
    print(result)
    print("test")
    preds = model.predict_proba(x_test)
    for i in range(0, 50):
        print("result=%.4f, %.4f" %(preds[i], y_test[i])) 
    result = metrics.roc_auc_score(y_test, preds)
    print(result)

def load_trend_model():
    model1 = model_from_json(open('trend_model.json').read())    
    model1.load_weights('trned_model_weights.h5')    

    model = Sequential()
    model.add(LSTM(input_shape=[15, 1],  output_dim =10, return_sequences = True, weights = model1.layers[0].get_weights()))
    model.add(Flatten(weights = model1.layers[1].get_weights()))
    model.add(Activation('linear', weights = model1.layers[2].get_weights()))
    model.add(Dense(output_dim=20, weights = model1.layers[3].get_weights()))
    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
def predict_trends(model, x_data):
    tmp_x = x_data[:, x_data.shape[1]-15:x_data.shape[1]]
    #tmp_x[:,1:] = tmp_x[:,1:]/tmp_x[:, 0:tmp_x.shape[1]-1]
    preds = model.predict_proba(tmp_x.reshape(x_data.shape[0],15,1))
    x_result = numpy.hstack((tmp_x, preds))
    print(preds[0:10])
    return x_result
if __name__ == "__main__":
    global sample_bdate
    global sample_edate
    global test_bdate
    global test_edate
    sample_bdate = sys.argv[1]
    sample_edate = sys.argv[2]
    test_bdate = sys.argv[3]
    test_edate = sys.argv[4]
    x_data, y_data, x_test, y_test = load_data("stock_sample_with_label_category")
    trend_model = load_trend_model()
    x_data1 = predict_trends(trend_model, x_data)
    x_test1 = predict_trends(trend_model, x_test)
    et = build_model(x_data1, y_data, x_test1, y_test)

