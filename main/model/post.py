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
        if os.path.exists(class_dump_file+".index") and not self.confer.force:
            if self.confer.classifier.get_name().startswith('ts'):
                self.confer.classifier.load(class_dump_file)
            else:
                with open(class_dump_file, 'rb') as fin:
                    print("load %s" % class_dump_file)
                    self.confer.classifier = pickle.load(fin)

                #self.confer.classifier.classifier = keras.models.load_model(class_dump_file)
        else:
            self._train(token.train, token.test, self.confer.scores[0])
            if self.confer.classifier.get_name().startswith("ts"):
                self.confer.classifier.save(class_dump_file)
            else:
                with open(class_dump_file, 'wb') as fout:
                    pickle.dump(self.confer.classifier, fout, protocol=-1)
                #self.confer.classifier.classifier.save(class_dump_file)


    def _train(self, df_train, df_test, score):
        df_train = df_train.sort_values(["sym", "date"])
        df_train_1 = df_train[df_train[score.get_name()] == 0]
        df_train_2 = df_train[df_train[score.get_name()] == 1]
        assert len(df_train_1) + len(df_train_2) == len(df_train)
        df_train_2 = df_train_2.sample(n = len(df_train_1))
        assert(len(df_train_2) == len(df_train_1))
        df_train = pd.concat([df_train_1, df_train_2], axis=0)
        assert(len(df_train) == 2*len(df_train_1))

        print("train start : %s train end: %s" % (df_train.sort_values('date').head(1)['date'].values[0],
                                                  df_train.sort_values('date').tail(1)['date'].values[0]))
        npTrainFeat, npTrainLabel = self._extract_feat_label(df_train, score.get_name())
        df_test = df_test.sort_values(["sym", "date"])
        npTestFeat, npTestLabel = self._extract_feat_label(df_test, score.get_name())
        self.confer.classifier.fit(npTrainFeat, npTrainLabel, npTestFeat, npTestLabel, npTestFeat, npTestLabel)
    def _extract_feat_label(self, df, scorename, drop = True):
        if drop:
            df = df.replace([np.inf,-np.inf],np.nan).dropna()
        feat_names = base.get_feat_names(df)
        npFeat = df.loc[:,feat_names].values.copy()
        npLabel = df.loc[:,scorename].values.copy()
        return npFeat, npLabel

    def pred(self, start = None):
        df_all = pd.read_pickle(self.confer.get_sel_file())
        if start != None:
            df_all = df_all[df_all.date >= start]
        feat_names = base.get_feat_names(df_all)
        np_feat = df_all.loc[:, feat_names].values
        np_pred = self.confer.classifier.predict_proba(np_feat)
        #df_all = df_all.iloc[2-1:]
        df_all["pred"] = np_pred[:, 1]
        df_all = df_all.sample(frac=1.0)
        return df_all.sort_values("pred", ascending=False)

    #def _mean(self, posts):
    #    mean_tpr = 0.0
    #    mean_fpr = np.linspace(0, 1, 100)
    #    for each in posts:
    #        mean_tpr += interp(mean_fpr, each.fpr, each.tpr)
    #        mean_tpr[0] = 0.0
    #    mean_tpr /= len(posts)
    #    mean_tpr[-1] = 1.0
    #    mean_auc = auc(mean_fpr, mean_tpr)
    #    return {"tpr":mean_tpr, "fpr":mean_fpr, "roc_auc":mean_auc}

