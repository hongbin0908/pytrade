#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys
import os
import numpy as np
import pandas as pd
import pickle
from scipy import interp
from sklearn.metrics import auc
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.work import conf
from main.model import model_work
from main.model import ana


class Poster:
    def __init__(self, classifier, ipts, fpr, tpr, thresholds,
            roc_auc, name, min, max, df_test):
        self.classifier = classifier
        self.ipts = ipts
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.roc_auc = roc_auc
        self.name = name
        self.min = min
        self.max = max
        self.df_test = df_test

    def accurate(self, threshold, score1, score2, top = 2):
        df_test = self.df_test.copy()
        df_test.sort_values("pred", ascending=False, inplace=True)
        df_select, df_year, df_month = ana.select2(score1, score2,
                                                    df_test,
                                                    top, threshold)
        return {
            "glo_l1": ana.accurate(df_test,    score1),
            "sel_l1": ana.accurate(df_select,  score1),
            "glo_l2": ana.accurate(df_test,    score2),
            "sel_l2": ana.accurate(df_select , score2),
            "sel_len": len(df_select),
            "min": float(df_select.tail(1)["pred"]) if len(df_select) > 0 else 0,
            "max": float(df_select.head(1)["pred"]) if len(df_select) > 0 else 0,
        }

class Crosser:
    def __init__(self, confer):

        self.posts = {"model":None, "valid":None}
        self.confer = confer
        self.symset = confer.syms
        assert isinstance(self.confer, conf.MltradeConf)
        self.posts = self._create( self.confer.model_split)

    def get_name(self):
        return self.symset.get_name()

    def get_mean(self, which):
        return self._mean(self.posts[which])

    def get_ta_file(self):
        return os.path.join(root, "data", "ta", "%s-%s-%s.pkl"
                            % (self.symset.get_name(), self.confer.ta.get_name(),
                               self.confer.score1.get_name()))

    def accurate(self, which, threshold):
        df_test = pd.concat([poster.df_test for poster in self.posts[which]])
        df_test.sort_values("pred", ascending=False, inplace=True)
        df_select, df_year, df_month = ana.select2( self.confer.score1, self.confer.score2,
                                                    df_test,
                                                    2, threshold)
        return {
            "symset": self.symset.get_name(),
            "glo_l1": ana.accurate(df_test,    self.confer.score1),
            "sel_l1": ana.accurate(df_select,  self.confer.score1),
            "glo_l2": ana.accurate(df_test,    self.confer.score2),
            "sel_l2": ana.accurate(df_select , self.confer.score2),
            "sel_len": len(df_select),
            "min": float(df_select.tail(1)["pred"]) if len(df_select) > 0 else 0,
            "max": float(df_select.head(1)["pred"]) if len(df_select) > 0 else 0,
        }


    def to_table(self, which):
        columns=["symset"]
        columns.extend(["cr_%d" % i for i in range(len(self.posts[which]))])
        columns.append("mean")
        data = [self.symset.get_name()]
        data.extend([each.roc_auc for each in self.posts[which]])
        data.append(self.get_mean(which)["roc_auc"])
        df = pd.DataFrame([data], columns=columns)
        return df

    def _mean(self, posts):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for each in posts:
            mean_tpr += interp(mean_fpr, each.fpr, each.tpr)
            mean_tpr[0] = 0.0
        mean_tpr /= len(posts)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        return {"tpr":mean_tpr, "fpr":mean_fpr, "roc_auc":mean_auc}

    def _create(self, split_):
        df_all = pd.read_pickle(self.confer.get_ta_file())
        token = split_.split(df_all)
        class_dump_file = os.path.join(root, 'data', 'cross', "%s-%s-%s-%s"
                                            % (self.confer.name, self.confer.syms.get_name(), token.train.name, self.confer.classifier.get_name()))
        tmp = token.train.sort_values(["date"])
        is_to_fit = True
        if os.path.exists(class_dump_file):
            with open(class_dump_file, 'rb') as fin:
                print("load %s" % class_dump_file)
                self.confer.classifier = pickle.load(fin)
                is_to_fit = False
        post =Poster(**model_work.post_valid(self.confer.classifier,
            token.train, token.test,
            self.confer.score1, is_to_fit))
        if is_to_fit:
            with open(class_dump_file, 'wb') as fout:
                pickle.dump(self.confer.classifier, fout, protocol=-1)
        return post

    def pred(self, start = None):
        df = pd.read_pickle(self.confer.get_ta_file())
        # df = df.replace([np.inf,-np.inf],np.nan).dropna()
        # today = df.sort_values("date", ascending=False)["date"].unique()[0]
        if start != None:
            df = df[df.date >= start]
        feat_names = base.get_feat_names(df)
        np_feat = df.loc[:, feat_names].values
        np_pred = self.posts.classifier.predict_proba(np_feat)
        df["pred"] = np_pred[:, 1]
        return df.sort_values("pred", ascending=False)


#class CrosserSet:
#    def __init__(self, confer):
#        self.confer = confer
#        self.crossers = []
#        for symset in confer.syms:
#            if not os.path.exists(self.get_dump_file(symset)):
#                print("%s Not exists, need to build!" % self.get_dump_file(symset))
#                crosser = Crosser(self.confer, symset)
#                with open(self.get_dump_file(symset), 'wb') as fout:
#                    pickle.dump(crosser, fout, protocol=-1)
#            else:
#                print("%s exists" % self.get_dump_file(symset))
#                with open(self.get_dump_file(symset), 'rb') as fin:
#                    crosser = pickle.load(fin)
#            self.crossers.append(crosser)
#
#    def get_dump_file(self, symset):
#        return os.path.join(root, 'data', 'cross', "%s-%s"
#                            % (self.confer.name, symset.get_name()))
#
#
#    def to_table(self, which):
#        df = pd.concat([each.to_table(which)
#                        for each in self.crossers])
#        df.reset_index(drop=True, inplace = True)
#        return df
#
#
#    def plot_roc(self, which, imgfile):
#        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
#        for crosser in self.crossers:
#            for color, post in zip(colors, crosser.posts[which]):
#                assert isinstance(post, Poster)
#                #plt.plot(each.fpr, each.tpr, lw=2, color=color, label=each.name)
#                plt.plot(post.fpr, post.tpr, lw=2, color=color)
#        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
#        model_work.plot_save(imgfile)
#
#    def _plot_precision_recall(self, which, imgfile, score_name, pos_label):
#        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
#        for crosser in self.crossers:
#            for color, post in zip(colors, crosser.posts[which]):
#                assert isinstance(post, Poster)
#                precision, recall, thresholds = precision_recall_curve(post.df_test[self.confer.score1.get_name()],
#                                       post.df_test[score_name], pos_label=pos_label)
#                s_scores = post.df_test[score_name].sort_values(ascending=False).reset_index(drop=True)
#                max_threshold = float(s_scores[100])
#                mask = (thresholds <= max_threshold)
#                mask = np.append(mask, False)
#                precision = precision[mask]
#                recall = recall[mask]
#                plt.plot(recall, precision, lw=0.5, color=color)
#        plt.xlim([-0.05, 1.05])
#        plt.ylim([-0.05, 1.05])
#        plt.xlabel('Recall')
#        plt.ylabel('Precision')
#        plt.title('Precision Recall which max threshold %f' % max_threshold)
#        plt.legend(loc="lower right")
#        plt.savefig(imgfile)
#        plt.cla()
#
#    def plot_precision_recall_bulls(self, which, imgfile):
#        """
#        :param which:
#        :param imgfile:
#        :return:
#        """
#        return self._plot_precision_recall(which, imgfile, score_name="pred", pos_label=1)
#
#    def plot_precision_recall_bears(self, which, imgfile):
#        """
#        :param which:
#        :param imgfile:
#        :return:
#        """
#        return self._plot_precision_recall(which, imgfile, score_name="pred2", pos_label=0)
#
#
#
#    def ipts_table(self, which):
#        df_ipt = None
#        for crosser in self.crossers:
#            assert isinstance(crosser, Crosser)
#            post = crosser.posts[which][0]
#            assert isinstance(post, Poster)
#            df = pd.DataFrame(post.ipts)
#            df.set_index("name", inplace=True)
#            if df_ipt is None:
#                df_ipt = df
#                df_ipt["score_all"] = df_ipt["score"]
#                df_ipt["score-%s" % crosser.symset.get_name()] = df_ipt["score"]
#                del df_ipt["score"]
#            else:
#                df_ipt = df_ipt.join(df)
#                df_ipt["score_all"] = df_ipt["score_all"] + df_ipt["score"]
#                df_ipt["score-%s" % crosser.symset.get_name()] = df_ipt["score"]
#                del df_ipt["score"]
#        df_ipt.sort_values("score_all", ascending=False, inplace=True)
#        df_ipt.reset_index(drop=False, inplace=True)
#        return df_ipt
#
#    def pred(self, start = None):
#        to_merged = []
#        for crosser in self.crossers:
#            assert isinstance(crosser, Crosser)
#            df = pd.read_pickle(crosser.get_ta_file())
#            # df = df.replace([np.inf,-np.inf],np.nan).dropna()
#            # today = df.sort_values("date", ascending=False)["date"].unique()[0]
#            if start != None:
#                df = df[df.date >= start]
#            feat_names = base.get_feat_names(df)
#            np_feat = df.loc[:, feat_names].values
#            np_pred = crosser.posts["valid"][0].classifier.predict_proba(np_feat)
#            df["pred"] = np_pred[:, 1]
#            to_merged.append(df)
#        return pd.concat(to_merged).sort_values("pred", ascending=False)
#
#    def plot_top_precision(self, which, imgfile, score_name, top = 10):
#        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
#        for crosser in self.crossers:
#            for color, post in zip(colors, crosser.posts[which]):
#                df_test = post.df_test.copy()
#                df_test.sort_values("pred", ascending=False, inplace=True)
#                df_select, df_year, df_month = ana.select2( self.confer.score1, 
#                    self.confer.score2, df_test, top, 100000, score_name)
#
#                precision, recall, thresholds = \
#                        precision_recall_curve(df_select[self.confer.score1.get_name()], 
#                                df_select["pred"])
#                plt.plot(thresholds, precision[1:], lw=2, color=color)
#        plt.xlabel('thresholds')
#        plt.ylabel('Precision')
#        plt.title('top presion')
#        plt.legend(loc="lower right")
#        plt.savefig(imgfile)
#        plt.cla()
#
#    def _top(self, which, score_name):
#        result = []
#        for crosser in self.crossers:
#            df_test = pd.concat([poster.df_test for poster in crosser.posts[which]])
#            df_test.sort_values(["date"])
#            df_select, df_year, df_month = ana.select2( self.confer.score1, self.confer.score2,
#                                                        df_test,
#                                                        10, 999999, score_name)
#            df_select['yyyy'] = df_select.date.str.slice(0, 4)
#            df = df_select.sort_values([score_name], ascending=False) \
#                .groupby('yyyy', as_index=False).head(2).sort_values( ['yyyy'])
#            df["which"] = which
#            cols = df.columns.tolist()
#            cols = cols[-1:] + cols[:-1]
#            df = df[cols]
#            result.append(df)
#        return pd.concat(result)[["which", "date", "yyyy", "sym", "pred", self.confer.score1.get_name()]]
#
#
#    def top_bulls(self, which):
#        return self._top(which, "pred")
#
#    def top_bears(self, which):
#        return self._top(which, "pred2")
#
#    def accurate(self, which):
#        result = []
#        for threshold in [2000, 1000, 500, 200, 100]:
#            for crosser in self.crossers:
#                assert isinstance(crosser, Crosser)
#                df_test = pd.concat([poster.df_test for poster in crosser.posts[which]])
#                df_test.sort_values("pred", ascending=False, inplace=True)
#                df_select, df_year, df_month = ana.select2( self.confer.score1, self.confer.score2,
#                                                            df_test,
#                                                            10, threshold, score_name="pred")
#                result.append({
#                    "symset": crosser.symset.get_name(),
#                    "glo_l1": ana.accurate(df_test,    self.confer.score1),
#                    "sel_l1": ana.accurate(df_select,  self.confer.score1),
#                    "glo_l2": ana.accurate(df_test,    self.confer.score2),
#                    "sel_l2": ana.accurate(df_select , self.confer.score2),
#                    "sel_len": len(df_select),
#                    "min": float(df_select.tail(1)["pred"]) if len(df_select) > 0 else 0,
#                    "max": float(df_select.head(1)["pred"]) if len(df_select) > 0 else 0,
#                })
#        return pd.DataFrame(result)
#
