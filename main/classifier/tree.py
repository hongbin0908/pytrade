#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np
np.random.seed(7)
import keras
from keras.layers import Flatten, Activation, Dense, Dropout
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras import initializers
from main.classifier.base_classifier import BaseClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model
import main.base as base

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


#from lasagne import layers  
#from lasagne.updates import nesterov_momentum  
#from nolearn.lasagne import NeuralNet 
#from nolearn.lasagne import visualize  
#import lasagne
#import pickle
from sklearn.metrics import confusion_matrix
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
from sklearn import metrics
import numpy as np
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM
#from keras.optimizers import SGD
#from keras.layers.core import Flatten

# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
import os, sys
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class cnn(BaseClassifier):
    def __init__(self, batch_size = 200, nb_epoch=100):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        pass
    def transfer_shape(self,X):
        return np.reshape(X, (X.shape[0], X.shape[1],1,1))
    def get_name(self):
        return "ccl-cnn"
    def fit(self, X, y, X_t, y_t):
        X = self.transfer_shape(X)
        X_t = self.transfer_shape(X_t)
        y = keras.utils.to_categorical(y, 2)
        y_t = keras.utils.to_categorical(y_t, 2)
        print(X.shape)
        if len(X_t) % 2==0:
            X_test, X_val = np.split(X_t, 2)
            y_test, y_val = np.split(y_t, 2)
        else:
            X_test, X_val = np.split(X_t[:-1,:], 2)
            y_test, y_val = np.split(y_t[:-1], 2)
        assert(len(X_test) == len(y_test))
        N = X.shape[0]
        Ntest = X_test.shape[0]
        D = X.shape[1]


        """Hyperparameters"""
        num_filt_1 = 6     #Number of filters in first conv layer
        num_filt_2 = 4      #Number of filters in second conv layer
        num_filt_3 = 8      #Number of filters in thirs conv layer
        num_fc_1 = 40       #Number of neurons in hully connected layer
        max_iterations = 20000
        dropout = 1.0       #Dropout rate in the fully connected layer
        plot_row = 5        #How many rows do you want to plot in the visualization
        learning_rate = 2e-5
        input_norm = False   # Do you want z-score input normalization?
        x = tf.placeholder("float", shape=[None, D], name = 'features')
        y_ = tf.placeholder(tf.int64, shape=[None], name = 'label')
        keep_prob = tf.placeholder("float")
        bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm
        #with tf.name_scope("Reshaping_data") as scope:
        #    x_image = tf.reshape(x, [-1,D,1,1])
        initializer = tf.contrib.layers.xavier_initializer()
        #with tf.name_scope("Conv1") as scope:
        #    W_conv1 = tf.get_variable("Conv_Layer_1",
        #                              shape=[5, 1, 1, num_filt_1],
        #                              initializer=initializer)
        #    b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
        #    a_conv1 = conv2d(x_image, W_conv1) + b_conv1
        self.classifier.add(Conv2D(filters=num_filt_1, kernel_size=[5,1], padding='same',
                                   kernel_initializer=initializer,
                                   input_shape=X.shape[1:]))
        #with tf.name_scope('Batch_norm_conv1') as scope:
        #    a_conv1 = tf.contrib.layers.batch_norm(a_conv1,is_training=bn_train,updates_collections=None)
        #    h_conv1 = tf.nn.relu(a_conv1)

        #model.add(BatchNormalization())
        self.classifier.add(Activation('relu'))

        #with tf.name_scope("Conv2") as scope:
        #    W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4, 1, num_filt_1, num_filt_2],initializer=initializer)
        #    b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
        #    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

        self.classifier.add(Conv2D(filters=num_filt_2, kernel_size=[4,1], kernel_initializer=initializer, padding='same'))


        #with tf.name_scope('Batch_norm_conv2') as scope:
        #    a_conv2 = tf.contrib.layers.batch_norm(a_conv2,is_training=bn_train,updates_collections=None)
        #    h_conv2 = tf.nn.relu(a_conv2)
        self.classifier.add(Activation('relu'))

        #with tf.name_scope("Fully_Connected1") as scope:
        #    W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D*num_filt_2, num_fc_1],initializer=initializer)
        #    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
        #    h_conv3_flat = tf.reshape(h_conv2, [-1, D*num_filt_2])
        #    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        self.classifier.add(Flatten())
        self.classifier.add(Dense(num_fc_1, kernel_initializer=initializer))
        self.classifier.add(Activation('relu'))

        #with tf.name_scope("Fully_Connected2") as scope:
        #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #    W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1,2],initializer=initializer)
        #    b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]),name = 'b_fc2')
        #    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.classifier.add(Dense(2, kernel_initializer=initializer, bias_initializer=initializers.Constant(value=0.1)))

        #with tf.name_scope("SoftMax") as scope:
        #    #    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
        #    #                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
        #    #                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
        #    #                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
        #    #                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2,labels=y_)
        # cost = tf.reduce_sum(loss) / batch_size
        self.classifier.add(Activation('softmax'))
        self.classifier.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate),matrics=['accuracy'])
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch)
         #    cost += regularization*regularizers
         #loss_summ = tf.summary.scalar("cross entropy_loss", cost)
         #3with tf.name_scope("train") as scope:
         #    tvars = tf.trainable_variables()
         #    #We clip the gradients to prevent explosion
         #    grads = tf.gradients(cost, tvars)
         #    optimizer = tf.train.AdamOptimizer(learning_rate)
         #    gradients = zip(grads, tvars)
         #    train_step = optimizer.apply_gradients(gradients)
         #    # The following block plots for every trainable variable
         #    #  - Histogram of the entries of the Tensor
         #    #  - Histogram of the gradient over the Tensor
         #    #  - Histogram of the grradient-norm over the Tensor
         #    numel = tf.constant([[0]])
         #    for gradient, variable in gradients:
         #        if isinstance(gradient, ops.IndexedSlices):
         #            grad_values = gradient.values
         #        else:
         #            grad_values = gradient

         #        numel +=tf.reduce_sum(tf.size(variable))

         #        h1 = tf.histogram_summary(variable.name, variable)
         #        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
         #        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
         #with tf.name_scope("Evaluating_accuracy") as scope:
         #    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
         #    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
         #    accuracy_summary = tf.summary.scalar("accuracy", accuracy)


         ##Define one op to call all summaries
         #merged = tf.summary.merge_all()

         #def print_tvars():
         #    tvars = tf.trainable_variables()
         #    for variable in tvars:
         #        print(variable.name)
         #    return
         #print_tvars()

         ## For now, we collect performances in a Numpy array.
         ## In future releases, I hope TensorBoard allows for more
         ## flexibility in plotting
         #perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
         #cost_ma = 0.0
         #acc_ma = 0.0
         #with tf.Session() as sess:
         #    writer = tf.summary.FileWriter(os.path.join(root, 'data', "tensorflow"), sess.graph_def)
         #    sess.run(tf.initialize_all_variables())
         #    step = 0      # Step is a counter for filling the numpy array perf_collect
         #    for i in range(max_iterations):
         #        batch_ind = np.random.choice(N,batch_size,replace=False)

         #        if i==0:
         #            # Use this line to check before-and-after test accuracy
         #            result = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
         #            acc_test_before = result
         #        if i%200 == 0:
         #            #Check training performance
         #            result = sess.run([cost,accuracy],feed_dict = { x: X, y_: y, keep_prob: 1.0, bn_train : False})
         #            perf_collect[1,step] = acc_train = result[1]
         #            cost_train = result[0]

         #            #Check validation performance
         #            result = sess.run([accuracy,cost,merged], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
         #            perf_collect[0,step] = acc_val = result[0]
         #            cost_val = result[1]
         #            if i == 0: cost_ma = cost_train
         #            if i == 0: acc_ma = acc_train
         #            cost_ma = 0.8*cost_ma+0.2*cost_train
         #            acc_ma = 0.8*acc_ma + 0.2*acc_train

         #            #Write information to TensorBoard
         #            writer.add_summary(result[2], i)
         #            writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
         #            print("At %5.0f/%5.0f Cost: train%5.3f val%5.3f(%5.3f) Acc: train%5.3f val%5.3f(%5.3f) " % (i,max_iterations, cost_train,cost_val,cost_ma,acc_train,acc_val,acc_ma))
         #            step +=1
         #        sess.run(train_step,feed_dict={x:X[batch_ind], y_: y[batch_ind], keep_prob: dropout, bn_train : True})
         #    result = sess.run([accuracy,numel], feed_dict={ x: X_t, y_: y_t, keep_prob: 1.0, bn_train : False})
         #    acc_test = result[0]
         #    print('The network has %s trainable parameters'%(result[1]))
    def predict_proba(self, X):
        X = self.transfer_shape(X)
        re = self.classifier.predict_proba(X)
        return re
def d2tod3(fro, window):
    row = fro.shape[0]
    feat_num = fro.shape[1]

    d1 = row - window +1
    d2 = window
    d3 = feat_num

    print(d1,d2,d3)
    to = np.zeros(d1*d2*d3).reshape(d1,d2,d3)
    for i in range(len(fro)-window + 1):
        to[i] = fro[i:i+window]
    return to
class ccl(BaseClassifier):
    def __init__(self, batch_size = 32, nb_epoch=10):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        pass
    def get_name(self):
        return "ccl-%d" % (self.nb_epoch)

    def transfer_shape(self,X):
        return d2tod3(X, window=2)
        return np.reshape(X, (X.shape[0], 1, X.shape[1]))

    def fit(self, X, y, X_t, y_t):
        X = self.transfer_shape(X)
        X_t = self.transfer_shape(X_t)
        y = y[2-1:]
        y_t = y_t[2-1:]
        #self.classifier.add(Dense(500, input_shape=( X.shape[1],)))
        self.classifier.add(LSTM(input_shape=(2, X.shape[2]),  output_dim =8,
                                 return_sequences = True ))
        self.classifier.add(Flatten())
        #self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dense( output_dim=8))
        self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.3, seed=7))
        self.classifier.add(Dense(output_dim=8))
        self.classifier.add(Activation('tanh'))
        self.classifier.add(Dense(output_dim=1))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        self.classifier.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch)
    def predict_proba(self, X):
        X = self.transfer_shape(X)
        re = self.classifier.predict_proba(X)
        re = np.hstack([1-re, re])
        return re

class MyLogisticRegressClassifier(BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self, C = 0.01, max_iter=2000):
        #self.classifier = linear_model.LogisticRegression(C=C, max_iter=2000, verbose=1, n_jobs = 30, tol=1e-6, solver='sag')
        #self.name = "lr-%f" % C
        self.classifier = linear_model.LogisticRegression(C=C, max_iter=2000, verbose=1, n_jobs = 30,  penalty='l2', tol = 1e-5)
        self.name = "lr-%f-%d" % (C,max_iter)
    def get_name(self):

        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.coef_[0]))
        return ipts

    def verify_predict(self, df):
        feat_names = base.get_feat_names(df)
        ipts = self.get_feature_importances(feat_names)
        s = 0
        for each in ipts:
            if int(df[each]) == 1 :
                s += ipts[each] * 1
        import math
        return 1 / (1 + math.exp(-1 * (s + self.classifier.intercept_)))

class MySGDClassifier(BaseClassifier):
    def __init__(self, n_iter=100):
        self.classifier = linear_model.SGDClassifier(alpha=0.01, loss='log', n_iter=n_iter, verbose=1)
        #self.classifier = linear_model.SGDClassifier(loss='modified_huber', n_iter=n_iter, verbose=1)
        self.name = "sgd-%d"%(n_iter) 
    def get_name(self):

        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.coef_[0]))
        return ipts
    def verify_predict(self, df):
        feat_names = base.get_feat_names(df)
        ipts = self.get_feature_importances(feat_names)
        
        s = None
        for each in ipts:
            tmp = df[each]*ipts[each]
            if s is None:
                s = tmp
            else:
                s += tmp
        return 1 / (1 + np.exp(-1 * (s + self.classifier.intercept_)))

class MyRandomForestClassifier(BaseClassifier):
    def __init__(self, verbose=1, n_estimators = 2000, max_depth=8, min_samples_leaf=10000,
                 n_jobs=40):
        self.classifier = RandomForestClassifier( **{'verbose': verbose,
                                                     'n_estimators': n_estimators,
                                                     'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                                                      'n_jobs': n_jobs})
        self.name = "rf_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )

    def get_name(self):
        return self.name

    def fit(self, X, y, X_t, y_t):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts


class MyRfClassifier(BaseClassifier):
    def __init__(self, n_estimators, max_depth, min_samples_leaf):
        self.classifier = RandomForestClassifier(**{'verbose':1, 'n_estimators': n_estimators,
                                                    'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,
                                                    'n_jobs':40})
        self.name = "rf_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )
    def get_name(self):
        return self.name

    def fit(self, X, y, X_t, y_t):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts
class RFCv1n2000md6msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 6
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md3msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 3
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md2msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 2
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n200md2msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 200
        max_depth = 2
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md6msl10000(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 6
        min_samples_leaf = 10000
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)


class MyGradientBoostingClassifier(BaseClassifier):
    def __init__(self, verbose=1, n_estimators=5, max_depth=6, min_samples_leaf=100):
        self.classifier = GradientBoostingClassifier( **{'verbose': verbose,
                                                     'n_estimators': n_estimators,
                                                     'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf
                                                     })
        self.name = "gb_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )

    def get_name(self):
        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts

class GBCv1n600md3lr001(MyGradientBoostingClassifier):
    def __init__(self):
        n_estimators = 600
        max_depth = 3
        learning_rate = 0.01
        self.classifier = GradientBoostingClassifier(**{'verbose': 1, 'n_estimators': n_estimators,
                                                        'max_depth': max_depth, 'learning_rate': learning_rate})
        self.name = "gb_n{n}_md{md}_lr{lr}".format(
            **{"n": n_estimators, "md": max_depth, "lr": learning_rate}
        )

class MyBayesClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = GaussianNB()
        self.name = "bayes"

    def get_name(self):
        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        return []
