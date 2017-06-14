#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np
import math

import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import os, sys
import tensorflow as tf

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.classifier.base_classifier import BaseClassifier
from main.classifier.interval_acc import IntervalAcc

class TfDnn(BaseClassifier):
    def __init__(self, dim = 64, hs = 3, batch_size = 100, nb_epoch=30, verbose = 1):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.dim = dim
        self.hs = hs
        pass
    def get_name(self):
        return "tf-dnn-%d-%d-%d-%d" % (self.nb_epoch, self.batch_size, self.dim, self.hs)
    def fit(self, X_train, y_train, df_test, score):
        self.input_dim = X_train.shape[1]
        with tf.Graph().as_default():
            self.x_pl = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="x")
            self.y_pl = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
            self.logits = self.inference(X_train)
            self.loss = self._loss(self.logits, self.y_pl)
            train_op = self.training(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            for epoch in range(self.nb_epoch):
                print("epoch %d" % epoch)
                i = 0; size = X_train.shape[0]
                while True:
                    end = i + self.batch_size
                    if end >= size:
                        end = size
                    _, loss_value = self.sess.run([train_op, self.loss], feed_dict={
                        self.x_pl:X_train[i:end], self.y_pl:y_train[i:end]
                    })
                    i = end
                    if i >= size:
                        break

    def inference(self, X):
        hidden = []
        with tf.name_scope('input'):
            weights = tf.Variable(
                tf.truncated_normal([self.input_dim, self.dim],
                                    stddev=1.0/math.sqrt(float(self.input_dim))),
                name = 'weights'
            )
            biases = tf.Variable(tf.zeros([self.dim]), name='biases')
            hidden.append(tf.nn.dropout(tf.nn.relu(tf.matmul(self.x_pl, weights) + biases), 0.5))

        for i in range(1, self.hs):
            with tf.name_scope('hidden%d' % i):
                weights = tf.Variable(
                    tf.truncated_normal([self.dim, self.dim],
                            stddev=1.0/math.sqrt(float(self.dim))),
                    name = 'weights'
                )
                biases = tf.Variable(tf.zeros([self.dim]), name='biases')
                hidden.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden[i-1], weights) + biases), 0.5))
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([self.dim, 1],
                                    stddev=1.0/math.sqrt(float(self.dim))),
                name='weights'
            )
            biases = tf.Variable(tf.zeros([1]), name='biases')

            logits = tf.nn.sigmoid(tf.matmul(hidden[i], weights) + biases, name="sigmoid")
        return logits

    def _loss(self, logits, labels):
        labels = tf.to_int64(labels)
        labels = tf.to_float(labels)
        return tf.nn.l2_loss(logits-labels, name = "squared_error_cost")
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def training(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        self.classifier.fit(X, y, shuffle=False, batch_size=self.batch_size, nb_epoch=self.nb_epoch, callbacks=[ival])


    def predict_proba(self, X):
        y = tf.nn.softmax(logits=self.logits)
        pred = self.sess.run(y, feed_dict={self.x_pl:X})
        return pred

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass
