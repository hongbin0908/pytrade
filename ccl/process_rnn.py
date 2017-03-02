import sys
import numpy
from lasagne import layers  
from lasagne.updates import nesterov_momentum  
from nolearn.lasagne import NeuralNet 
from nolearn.lasagne import visualize  
import lasagne
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
#from keras.layers.recurrent import SimpleDeepRNN

sample_bdate = ''
sample_edate = ''
test_bdate = ''
test_edate = ''
interval = 15
def weighted_crossentropy(predictions, targets):
    print(predictions.shape)
    print(targets.shape)
    loss = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(predictions, targets))
    return loss
def get_sample(data_info, sample_bdate, sample_edate, num_need,m_input, m3_input):
    global interval
    try:
        x_data = data_info[data_info.date>sample_bdate]
        x_data = x_data[x_data.date<sample_edate].iloc[:, [0,1,2,3,4,5,6, 7]]
        m = x_data.sort(['sym', 'date']).groupby(['sym'])
        m1 = m_input
        m3 = m3_input
        for name, group in m:
            t = group[['volume']]
            print(t.shape)
            print(num_need)
            if t.shape[0] < num_need or t.shape[0] < interval:
                continue
            t1 = t[0:int(t.shape[0]/interval)*interval].values.reshape(int(t.shape[0]/interval), interval)
            if m1 == "":
                m1 = t1
            else:
                m1 = np.vstack((m1,t1))
            t2 = group['score_label_1_100'][0:int(t.shape[0]/interval)*interval].values.reshape(int(t.shape[0]/interval),interval)
            if m3 == "":
                m3 = t2
            else:
                m3 = np.vstack((m3, t2))
        print("m1=")
        if m1=="":
            return None, None, None, "", ""
        print(m1)
        m2 = m1/m1[:,0].reshape(m1.shape[0],1)
        #m2[1:45] = m2[1:45]
        return m2[:,1:interval].reshape(-1, interval-1, 1)*10-10, m3[:,interval-1], m3[:,interval-1], m1, m3
    except Exception as e:
             print("tmp info = %s" %(e))
             sys.exit(1)
def load_data(filename):
    global sample_bdate
    global sample_edate
    global test_bdate
    global test_edate
    try:
        fd = open(filename, "rb")
        data_info = pickle.load(fd)
        print(data_info.head())
        x_data, y_data, x_result, m1, m3 = get_sample(data_info, sample_bdate, sample_edate, 252, "", "")
        x_test, y_test, r_test, m1, m3 = get_sample(data_info, test_bdate, test_edate, interval, "", "")
        return data_info, x_data, y_data, x_result, x_test, y_test, r_test
    except Exception as e:
        print("info = %s" %(e))
        sys.exit(1)
def build_model(data_info, x_data, y_data, y_result, x_test, y_test, r_test):
    print(x_data[0])
    print(y_data[0])
    model = Sequential()
    model.add(LSTM(input_shape=[x_data.shape[1], 1],  output_dim =10, return_sequences = True))
    model.add(Flatten())
    model.add(Activation('linear'))
    model.add(Dense(output_dim=20))
    model.add(Activation('linear'))
    model.add(Dense(output_dim=10))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    model.fit(x_data, y_data, batch_size=10000, nb_epoch=20, validation_split=0.2)
    i = 0
    while i<=150:
        try:
            global test_bdate
            global test_edate
            bint = int(test_bdate.split("-")[0])
            eint = int(test_edate.split("-")[1])
            byear = random.randint(bint-10, bint-1)
            bmoth = random.randint(1,12)
            eyear = random.randint(byear, bint-1)
            emoth = random.randint(1,12)
            bdate = str(byear) + "-" + str(bmoth)+ "-01"
            edate = str(eyear) + "-" + str(emoth) + "-01"
            x1, y1, t1, m1, m3 = get_sample(data_info, bdate, edate, interval, "", "")
            if m1 == "" or x1==None:
                i =i + 1
                continue
            preds = model.predict_proba(x1)
            result = metrics.roc_auc_score(t1, preds)
            print("before adjust, result = %f, bdate=%s, edate=%s" %(result, bdate, edate))
            if result<0.52:
                batch_num = int(x1.shape[0]/9)
                if batch_num > 500:
                    batch_num = 500
                model.fit(x1, y1, batch_size = batch_num, nb_epoch = 10, validation_split = 0.2)
                preds = model.predict_proba(x1)
                result = metrics.roc_auc_score(t1, preds)
                print("after adjust, result = %f, bdate=%s, edate=%s" %(result, bdate, edate))
            i = i+1
        except Exception as e:
            print("error information=%s" %(e))
            continue
    score = model.evaluate(x_data, y_data, batch_size=16)
    preds = model.predict_proba(x_data)
    print(preds[0:10])
    print(y_data[0:10])
    print(x_data[0:10])
#    pred1 = model.predict_classes(x_data)
#    print(score)
#    j = 0
#    for i in preds:
#        if i>0.7:
#            j = j +1
    
#    cm = confusion_matrix(y_data, pred1) 
#    print(cm)
    result = metrics.roc_auc_score(y_result, preds)
    print(result)

    print("test")
    preds = model.predict_proba(x_test)
    for i in range(0, preds.shape[0]):
        print("result=%.4f, %.4f" %(preds[i], r_test[i])) 
    print(preds[0:10])
    print(y_test[0:10])
    result = metrics.roc_auc_score(r_test, preds)
    print(result)
#    score = model.evaluate(X_test, Y_test, batch_size=16)
#    l_shp = layers.ReshapeLayer(l_lstm, (-1, 10))
#    l_dense = layers.DenseLayer(input_layer, num_units=6)
#    l_dense1 = layers.DenseLayer(incoming=l_dense,num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
#    et1 = NeuralNet(layers = [input_layer, l_dense, l_lstm], update=nesterov_momentum,  update_learning_rate=0.1, update_momentum=0.3, max_epochs=50,regression = True, 
 #          verbose=1, )
#    et1 = NeuralNet(  
#        layers=[('input', layers.InputLayer),  
#        ('dense1', layers.DenseLayer),   
##        ('dropout2', layers.DropoutLayer),
#        ('dense', layers.DenseLayer),  
#        ('output', layers.DenseLayer),], 
#        input_shape=(None, x_data.shape[1]), 
#        dense_num_units=30,
#        dense1_num_units=30,
#        dense_nonlinearity=lasagne.nonlinearities.rectify,
#        dense1_nonlinearity=lasagne.nonlinearities.rectify,
##        dropout2_p=0.01,
#        output_nonlinearity=lasagne.nonlinearities.softmax,  
#        output_num_units=2,
#        update=nesterov_momentum,  
#        update_learning_rate=0.1,  
#        update_momentum=0.3,  
#        max_epochs=50,  
#        verbose=1,  )
#    et1.fit(x_data, y_data)
#    return et1

def test_model(x_data, y_data, et):
    preds = et.predict(x_data) 
#    print(lasagne.layers.get_output(et))
#    cm = confusion_matrix(y_data, preds) 
#    print(cm)
    result = metrics.roc_auc_score(y_data, preds)
    print(result)
#    cm = confusion_matrix(y_test, preds)  
#    plt.matshow(cm)  
#    plt.title('Confusion matrix')  
#    plt.colorbar()  
#    plt.ylabel('True label')  
#    plt.xlabel('Predicted label')  
#    plt.show() 

if __name__ == "__main__":
    global sample_bdate
    global sample_edate
    global test_bdate
    global test_edate
    sample_bdate = sys.argv[1]
    sample_edate = sys.argv[2]
    test_bdate = sys.argv[3]
    test_edate = sys.argv[4]
    data_info, x_data, y_data, y_result, x_test, y_test, r_test = load_data("./data/ta/sp500_snapshot_20091231-TaBase1Ext4El-0-score_label_1_100.pkl")
#    x_data1, y_data1, y_result1, x_test1, y_test1, r_test1 = load_data("sp100w150i0-TaBase1Ext4El-score_label_5_100.pkl")
    et = build_model(data_info, x_data, y_data, y_result, x_test, y_test, r_test)
#    test_model(x_test, y_test, et)

