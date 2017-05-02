import sys
import pandas
import numpy
import pickle
from sklearn import preprocessing
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
import datetime
sample_bdate = ''
sample_edate = ''
test_bdate = ''
test_edate = ''
def weighted_crossentropy(predictions, targets):
    loss = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(predictions, targets))
    return loss
def get_sample(data_info, sort_list, day_interval, day_interval1):
    global interval
    date_list = []
    if len(sort_list) > 50:
        length = 50
    else:
        length = len(sort_list)
    for i in range(0,length):
        date_list.append(sort_list[i][0])
    print(date_list[0:10])
    v1 = ""
    y1 = ""
    v2 = ""
    try:
        for i in date_list:
            b_date = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(days=-day_interval)
            b_date_str = b_date.strftime("%Y-%m-%d")
            e_date = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(days=day_interval1)
            e_date_str = e_date.strftime("%Y-%m-%d") 
            x_data = data_info[data_info.date>=b_date_str]
            x_data = x_data[x_data.date<=e_date_str]
            x_data2 = x_data.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
            x_data3= x_data2/x_data2[:, 0].reshape(x_data.shape[0],1)
            x_data4 = numpy.array(x_data3)
            x_data4[:,1:] = x_data4[:,1:]/x_data4[:,0:x_data4.shape[1]-1]
            y_data = data_info[data_info.date>=b_date_str]
            y_data = y_data[y_data.date<=e_date_str]
            y_data = y_data.iloc[:, 17].values.reshape(y_data.shape[0], 1)
            if v1 == "":
                v1 = x_data3
                y1 = y_data
                v2 = x_data4
            else:
                v1 = numpy.vstack((v1, x_data3))
                y1 = numpy.vstack((y1, y_data))
                v2 = numpy.vstack((v2, x_data4))
        return v1, v2, y1
    except Exception as e:
             print("get_sample info = %s" %(e))
             sys.exit(1)

def load_data(filename, sort_list, test_bdate):
    try:
        print(test_bdate)
        e_date = datetime.datetime.strptime(test_bdate[0], "%Y-%m-%d") + datetime.timedelta(days=-1)
        e_date_str = e_date.strftime("%Y-%m-%d") 
        sort_list.append((e_date_str, 10, 0))
        data_info = pandas.read_csv(filename, sep = ' ', header=0, names=['sym', 'date', 'price15', 'price14', \
                'price13', 'price12', 'price11', 'price10', 'price9', 'price8', 'price7', 'price6', 'price5', 'price4', 'price3','price2', 'price1', 'label'])
        x_data, x_dev, y_data = get_sample(data_info, sort_list, 2, 2)
        test_list = [(test_bdate[0], 0, 0)]
        x_test, x_test_dev, y_test = get_sample(data_info, test_list,0, 0)
        return x_data, x_dev, y_data, x_test, x_test_dev, y_test
    except Exception as e:
        print("info = %s" %(e))
        sys.exit(1)

def load_sp(filename, test_bdate, test_edate):
    try:
        data_info = pandas.read_csv(filename, sep = ' ', header=0, names=['sym', 'date', 'date_select', 'price15', 'price14', \
                'price13', 'price12', 'price11', 'price10', 'price9', 'price8', 'price7', 'price6', 'price5', 'price4', 'price3','price2', 'price1', 'label'])
        x_test = data_info[data_info.date_select==test_edate].values
        x_data = data_info[data_info.date_select<x_test[0,1]]
        x_data = x_data[data_info.date_select>"2007-01-01"].values
        if x_test.shape[0] == 0:
            print("the test data is empty")
            sys.exit(1)
        return x_data.reshape(x_data.shape[0], x_data.shape[1], 1), x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    except Exception as e:
        print("load_sp: info = %s" %(e))
        sys.exit(1)

def get_vector_sp(model, x_data, x_test):
    tmp_x = x_data[:, x_data.shape[1]-16:x_data.shape[1]-1]
    tmp_x = tmp_x.reshape(tmp_x.shape[0], tmp_x.shape[1])
    tmp_x = tmp_x/tmp_x[:,0].reshape(tmp_x.shape[0],1)
    #tmp_x[:,1:] = tmp_x[:,1:] / tmp_x[:,0:tmp_x.shape[1]-1]
    tmp_x = preprocessing.normalize(tmp_x, norm='l2')
    preds = model.predict_proba(tmp_x.reshape(x_data.shape[0],15,1))


    test_x = x_test[:, x_test.shape[1]-16:x_test.shape[1]-1]
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1])
    test_x = test_x/test_x[0][0]
    test_x = preprocessing.normalize(test_x, norm='l2')
    pred_test = model.predict_proba(test_x.reshape(test_x.shape[0],15,1))
    x_result = numpy.hstack((x_data.reshape(x_data.shape[0], x_data.shape[1]), preds))
    x_result_test = numpy.hstack((x_test.reshape(x_test.shape[0], x_test.shape[1]), pred_test))
    test_bdate = x_test[0][1]
    tmp_list = []
    tmp_vec = x_result_test[0][x_result_test.shape[1]-20:]
    i  = 0
    for m in x_result:
        i = i + 1
        dist = numpy.sqrt(numpy.sum(numpy.square(m[m.shape[0]-20:]- pred_test[0])))
        
        tmp_list.append((m[1], dist))
    sort_list = sorted(tmp_list, key = lambda x:x[1], reverse =False)
    return sort_list, test_bdate

#    return x_result

def build_model(x_data, y_data, x_test, y_test):
    model = Sequential()
    model.add(Dense(input_shape=[x_data.shape[1]],output_dim=1000, kernel_initializer= initializers.RandomNormal()))
    model.add(Activation('tanh'))
#    model.add(Dropout(0.4))
#    model.add(Activation('linear'))
    model.add(Dense(output_dim=1000, kernel_initializer= initializers.RandomNormal()))
    model.add(Activation('tanh'))
#    model.add(Dense(output_dim=100))
#    model.add(Activation('relu'))
    model.add(Dense(output_dim=1, kernel_initializer= initializers.RandomNormal()))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.005, decay=0.0001, momentum=0.2, nesterov=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)
#    model.fit(x_data, y_data, batch_size=4, nb_epoch=20, shuffle=True, validation_split=0.2)
    model.fit(x_data, y_data)
    preds = model.predict_proba(x_data)
    result = metrics.roc_auc_score(y_data, preds)
    print(result)
    print("test")
    preds = model.predict_proba(x_test)
    for i in range(0, 10):
        print("result=%.4f, %.4f" %(preds[i], y_test[i])) 
    result = metrics.roc_auc_score(y_test, preds)
    print(result)
    num = [0.0001, 0, 0]
    res_list = numpy.hstack((y_test, preds))
    res_sorted = sorted(res_list, key = lambda x:x[1], reverse = True)
    for i in range(0, y_test.shape[0]):
        if  i<=3 and res_sorted[i][1] >= 0.5:
            print("single result = %s" %(res_sorted[i]))
            num[0] += 1
        if res_sorted[i][1]>=0.5 and res_sorted[i][0]== 1  and i<=3:
            num[1] += 1
        if y_test[i] == 1:
           num[2] += 1
    for i in model.model.metrics_names:
        print("gradient update = %s" %(i))
    print("process_result: %.4f %d %d %d %.4f" %(num[0], num[1], num[2], preds.shape[0], num[1]/num[0]))
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
def predict_trends(model, x_data, x_dev):
    print("_______________________")
    print(x_data[0:2])
    print("_______________________")
    print(x_dev[0:2])
    tmp_x = x_data[:, x_data.shape[1]-15:x_data.shape[1]]
#    tmp_x = preprocessing.normalize(tmp_x, norm='l2')
#    tmp_x[:,1:] = tmp_x[:,1:]/tmp_x[:, 0:tmp_x.shape[1]-1]
    preds = model.predict_proba(tmp_x.reshape(x_data.shape[0],15,1))
#    preds = preprocessing.normalize(preds, norm='l2')
    dev_x = x_dev[:, x_dev.shape[1]-15:x_dev.shape[1]]
#    dev_x = preprocessing.normalize(dev_x, norm='l2')
    #dev_x[:,1:] = dev_x[:,1:]/dev_x[:, 0:dev_x.shape[1]-1]
    preds_dev = model.predict_proba(dev_x.reshape(x_dev.shape[0],15,1))
#    preds_dev = preprocessing.normalize(preds_dev, norm='l2')
    print(x_data.shape)
    print(preds.shape)
    print(preds_dev.shape)
    x_result = numpy.hstack((x_data, preds, preds_dev))
#    x_result = numpy.hstack((x_data, preds))
    print(x_result[0:10])
    return x_result


if __name__ == "__main__":
    test_edate = sys.argv[1]
    trend_model = load_trend_model()
    sp_data, sp_test = load_sp("spc_data", test_bdate, test_edate)
    sort_list, test_bdate = get_vector_sp(trend_model, sp_data, sp_test)
#    sort_list = []
    x_data, x_dev, y_data, x_test, x_test_dev, y_test = load_data("stock_sample_with_label_part", sort_list, test_bdate)
    x_data1 = predict_trends(trend_model, x_data, x_dev)
    x_test1 = predict_trends(trend_model, x_test, x_test_dev)
#    x_data1 = preprocessing.normalize(x_data1, norm='l2')
#    x_test1 = preprocessing.normalize(x_test1, norm='l2')
    et = build_model(x_data1, y_data, x_test1, y_test)

