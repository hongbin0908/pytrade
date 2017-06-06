#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys
import os
import numpy as np
import pandas as pd
import pickle
import keras

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.work import conf
from main.model import model_work


class Poster:
    def __init__(self, confer):
        self.confer = confer

    def get_name(self):
        return self.confer.syms.get_name()
    
    def work(self):
        df_all = pd.read_pickle(self.confer.get_sel_file())
        df_all.reset_index(drop=True, inplace=True)

        token = self.confer.model_split.split(df_all)
        class_dump_file = self.confer.get_classifier_file()
        tmp = token.train.sort_values(["date"])
        is_to_fit = True

        if self.confer.classifier.get_name().startswith("ts"):
            self.confer.classifier.init_cnn(D=self._extract_feat_label(df_all, self.confer.scores[0].get_name())[0].shape[1], num_class=2)
        if os.path.exists(class_dump_file) and not self.confer.force:
            if self.confer.classifier.get_name().startswith('ts'):
                self.confer.classifier.load(class_dump_file)
            elif self.confer.classifier.get_name().startswith('ccl'):
                self.confer.classifier.classifier = keras.models.load_model(class_dump_file)
            else:
                with open(class_dump_file, 'rb') as fin:
                    print("load %s" % class_dump_file)
                    self.confer.classifier = pickle.load(fin)
        else:
            if self.confer.classifier.get_name().startswith('mdn'):
                self._train(token.train, token.test, self.confer.scores[0])
            else:
                self._train(token.train, token.test, self.confer.scores[0])
            if self.confer.classifier.get_name().startswith("ts"):
                self.confer.classifier.save(class_dump_file)
            elif self.confer.classifier.get_name().startswith('ccl'):
                self.confer.classifier.save(class_dump_file)
            elif self.confer.classifier.get_name().startswith('mdn'):
                return
            else:
                with open(class_dump_file, 'wb') as fout:
                    pickle.dump(self.confer.classifier, fout, protocol=-1)


    def _train(self, df_train, df_test, score):
        df_train = df_train.sort_values(["sym", "date"])

        df_train_1 = df_train[df_train[score.get_name()] < 0.5]

        #df_train_1 = df_train_1.sort_values([self.confer.scores[1].get_name()], ascending=True).head(int(len(df_train_1)/3*2)).tail(int(len(df_train_1)/3))
        #print(df_train_1.head()[["sym", "date", "close", self.confer.scores[1].get_name()]])
        df_train_2 = df_train[df_train[score.get_name()] > 0.5]
        #df_train_2 = df_train_2.sort_values([self.confer.scores[1].get_name()], ascending=False).head(int(len(df_train_2)/3*2)).tail(int(len(df_train_2)/3))
        #print(df_train_2.head()[["sym", "date", "close", self.confer.scores[1].get_name()]])

        # @ccl
        df_train_2 = df_train_2.sample(n = len(df_train_1))
        assert(len(df_train_2) == len(df_train_1))
        df_train = pd.concat([df_train_1, df_train_2], axis=0)
        df_train = df_train.sample(frac=1.0)
        assert(len(df_train) == 2*len(df_train_1))

        print("train start : %s train end: %s total:%d" % (df_train.sort_values('date').head(1)['date'].values[0],
                                                  df_train.sort_values('date').tail(1)['date'].values[0], len(df_train)))
        npTrainFeat, npTrainLabel = base.extract_feat_label(df_train, score.get_name())
        df_test = df_test.sort_values(["sym", "date"])
        df_test_1 = df_test[df_test[score.get_name()] < 0.5]
        df_test_2 = df_test[df_test[score.get_name()] > 0.5]
#        assert len(df_test_1) + len(df_test_2) == len(df_test)
        df_test_2 = df_test_2.sample(n = len(df_test_1))
        assert(len(df_test_2) == len(df_test_1))
        df_test = pd.concat([df_test_1, df_test_2], axis=0)
        assert(len(df_test) == 2*len(df_test_1))
        df_test = df_test.sample(frac=1.0, random_state = 1253)
        npTestFeat, npTestLabel = base.extract_feat_label(df_test, score.get_name())
        #self.confer.classifier.fit(npTrainFeat, npTrainLabel, npTestFeat, npTestLabel, npTestFeat, npTestLabel)
        #self.confer.classifier.fit(npTrainFeat, npTrainLabel, npTestFeat, npTestLabel)
        self.confer.classifier.fit(npTrainFeat, npTrainLabel, df_test, score.get_name())

    def pred(self, start = None):
        df_all = pd.read_pickle(self.confer.get_sel_file())
        if start != None:
            df_all = df_all[df_all.date >= start]
        score = self.confer.scores[0]
        df_all_1 = df_all[df_all[score.get_name()] < 0.5]
        df_all_2 = df_all[df_all[score.get_name()] > 0.5]
        assert len(df_all_1) + len(df_all_2) == len(df_all)
        df_all_2 = df_all_2.sample(n = len(df_all_1))
        assert(len(df_all_2) == len(df_all_1))
        df_all = pd.concat([df_all_1, df_all_2], axis=0)
        assert(len(df_all) == 2*len(df_all_1))
        df_all = df_all.sample(frac=1.0, random_state = 1253)
        feat_names = base.get_feat_names(df_all)
        np_feat = df_all.loc[:, feat_names].values
        print("pred start : %s pred end: %s total:%d" % (df_all.sort_values('date').head(1)['date'].values[0],
                                                           df_all.sort_values('date').tail(1)['date'].values[0], len(df_all)))
        np_pred = self.confer.classifier.predict_proba(np_feat)
        #df_all = df_all.iloc[2-1:]
        if np_pred.shape[1] == 2:
            df_all["pred"] = np_pred[:, 1]
        else:
            df_all["pred"] = np_pred[:, 0]
        df_all = df_all.sample(frac=1.0)
        return df_all.sort_values("pred", ascending=False)


class ConPoster():
    def __init__(self, confer):
        self.confer = confer

    def get_name(self):
        return self.confer.syms.get_name()

    def work(self):
        df_all = pd.read_pickle(self.confer.get_score_with_original_file())
        df_all.reset_index(drop=True, inplace=True)

        token = self.confer.model_split.split(df_all)
        class_dump_file = self.confer.get_classifier_file()
        tmp = token.train.sort_values(["date"])
        is_to_fit = True

        if self.confer.classifier.get_name().startswith("ts"):
            self.confer.classifier.init_cnn(
                D=self._extract_feat_label(df_all, self.confer.scores[0].get_name())[0].shape[1], num_class=2)
        if os.path.exists(class_dump_file) and not self.confer.force:
            if self.confer.classifier.get_name().startswith('ts'):
                self.confer.classifier.load(class_dump_file)
            elif self.confer.classifier.get_name().startswith('ccl'):
                self.confer.classifier.classifier = keras.models.load_model(class_dump_file)
            else:
                with open(class_dump_file, 'rb') as fin:
                    print("load %s" % class_dump_file)
                    self.confer.classifier = pickle.load(fin)
        else:
            if self.confer.classifier.get_name().startswith('mdn'):
                self._train(token.train, token.test, self.confer.scores[0])
            else:
                self._train(token.train, token.test, self.confer.scores[0])
            if self.confer.classifier.get_name().startswith("ts"):
                self.confer.classifier.save(class_dump_file)
            elif self.confer.classifier.get_name().startswith('ccl'):
                self.confer.classifier.save(class_dump_file)
            elif self.confer.classifier.get_name().startswith('mdn'):
                return
            else:
                with open(class_dump_file, 'wb') as fout:
                    pickle.dump(self.confer.classifier, fout, protocol=-1)

    def _train(self, df_train, df_test, score):
        df_train = df_train.sort_values(["sym", "date"])

        df_train_1 = df_train[df_train[score.get_name()] < 0.5]

        # df_train_1 = df_train_1.sort_values([self.confer.scores[1].get_name()], ascending=True).head(int(len(df_train_1)/3*2)).tail(int(len(df_train_1)/3))
        # print(df_train_1.head()[["sym", "date", "close", self.confer.scores[1].get_name()]])
        df_train_2 = df_train[df_train[score.get_name()] > 0.5]
        # df_train_2 = df_train_2.sort_values([self.confer.scores[1].get_name()], ascending=False).head(int(len(df_train_2)/3*2)).tail(int(len(df_train_2)/3))
        # print(df_train_2.head()[["sym", "date", "close", self.confer.scores[1].get_name()]])

        # @ccl
        df_train_2 = df_train_2.sample(n=len(df_train_1))
        assert (len(df_train_2) == len(df_train_1))
        df_train = pd.concat([df_train_1, df_train_2], axis=0)
        df_train = df_train.sample(frac=1.0)
        assert (len(df_train) == 2 * len(df_train_1))

        print("train start : %s train end: %s total:%d" % (df_train.sort_values('date').head(1)['date'].values[0],
                                                           df_train.sort_values('date').tail(1)['date'].values[0],
                                                           len(df_train)))
        npTrainFeat, npTrainLabel = base.extract_feat_label(df_train, score.get_name())
        df_test = df_test.sort_values(["sym", "date"])
        df_test_1 = df_test[df_test[score.get_name()] < 0.5]
        df_test_2 = df_test[df_test[score.get_name()] > 0.5]
        #        assert len(df_test_1) + len(df_test_2) == len(df_test)
        df_test_2 = df_test_2.sample(n=len(df_test_1))
        assert (len(df_test_2) == len(df_test_1))
        df_test = pd.concat([df_test_1, df_test_2], axis=0)
        assert (len(df_test) == 2 * len(df_test_1))
        df_test = df_test.sample(frac=1.0, random_state=1253)
        npTestFeat, npTestLabel = base.extract_feat_label(df_test, score.get_name())
        # self.confer.classifier.fit(npTrainFeat, npTrainLabel, npTestFeat, npTestLabel, npTestFeat, npTestLabel)
        # self.confer.classifier.fit(npTrainFeat, npTrainLabel, npTestFeat, npTestLabel)
        self.confer.classifier.fit(npTrainFeat, npTrainLabel, df_test, score.get_name())

    def pred(self, start=None):
        df_all = pd.read_pickle(self.confer.get_score_with_original_file())
        if start != None:
            df_all = df_all[df_all.date >= start]
        score = self.confer.scores[0]
        df_all_1 = df_all[df_all[score.get_name()] < 0.5]
        df_all_2 = df_all[df_all[score.get_name()] > 0.5]
        assert len(df_all_1) + len(df_all_2) == len(df_all)
        df_all_2 = df_all_2.sample(n=len(df_all_1))
        assert (len(df_all_2) == len(df_all_1))
        df_all = pd.concat([df_all_1, df_all_2], axis=0)
        assert (len(df_all) == 2 * len(df_all_1))
        df_all = df_all.sample(frac=1.0, random_state=1253)
        feat_names = base.get_feat_names(df_all)
        np_feat = df_all.loc[:, feat_names].values
        print("pred start : %s pred end: %s total:%d" % (df_all.sort_values('date').head(1)['date'].values[0],
                                                         df_all.sort_values('date').tail(1)['date'].values[0],
                                                         len(df_all)))
        np_pred = self.confer.classifier.predict_proba(np_feat)
        # df_all = df_all.iloc[2-1:]
        if self.confer.classifier.get_name().startswith('mdn'):
            df_all["pred"] = np_pred[:, 0]
            df_all['threshold'] = np_pred[:,1]
        else:
            df_all["pred"] = np_pred[:, 1]
        df_all = df_all.sample(frac=1.0)
        return df_all.sort_values("pred", ascending=False)



