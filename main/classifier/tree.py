#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

from main.classifier.base_classifier import BaseClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model
import main.base as base


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
class ccl(BaseClassifier):
    def __init__(self):
        pass
    def get_name(self):
        return "ccl"
    def fit(self, X, y):
        model = Sequential()
        model.add(LSTM(input_shape=[X.reshap(-1, 118,1).shape[1], 1],  output_dim =30, return_sequences = True))
        model.add(Flatten())
        model.add(Activation('linear'))
        model.add(Dense( output_dim=30))
        model.add(Activation('linear'))
        model.add(Dropout(0.3))
        model.add(Dense(output_dim=10))
        model.add(Activation('tanh'))
        model.add(Dense(output_dim=1))
        model.add(Activation('sigmoid'))
        sgd = SGD(lr=0.05, decay=1e-5, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer='sgd')

        model.fit(X, y, batch_size=200, nb_epoch=100)
    def predict_proba(self, X):
        return model.predict_proba(X)

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

    def fit(self, X, y):
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

    def fit(self, X, y):
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
