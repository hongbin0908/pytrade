import os, sys
import pandas as pd
from keras.callbacks import Callback
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)
import main.base as base
class IntervalAcc(Callback):
    def __init__(self, cls, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.cls = cls
        self.df_test_valid, self.score = validation_data
        #self.df_test = self.df_test_valid.sample(frac=0.5, random_state=200)
        self.df_test_valid = self.df_test_valid.sort_values("date", ascending=True)
        self.df_test = self.df_test_valid.head(int(len(self.df_test_valid)/2))
        self.df_valid = self.df_test_valid.drop(self.df_test.index)
        assert(len(self.df_valid) + len(self.df_test) == len(self.df_test_valid))

        self.df_years = [each for each in base.split_by_year(self.df_test_valid)]

        self.npFeatTest, self.npLabelTest = base.extract_feat_label(self.df_test, self.score, drop=True)
        self.npFeatVal, self.npLabelVal = base.extract_feat_label(self.df_valid, self.score, drop=True)

    def cal_accuracy(self, npFeat, npLabel, is_short = False):
        y_pred = self.cls.predict_proba(npFeat)
        if is_short:
            df = pd.DataFrame({"pred": y_pred[:,0], "val": 1-npLabel})
        else:
            df = pd.DataFrame({"pred": y_pred[:,1], "val": npLabel})
        df.sort_values(["pred"], ascending=False, inplace=True)
        df1 = df.head(1000)
        score1 = len(df1[df1.val == 1])/len(df1)
        threshold1 = float(df1.tail(1)["pred"].values)
        df2 = df.head(10000)
        score2 = len(df2[df2.val == 1])/len(df2)
        threshold2 = float(df2.tail(1)["pred"].values)
        dfn = df[df.pred >= 0.5]
        if len(dfn) == 0:
            thresholdn = 0.5
            scoren = 0.0
        else:
            scoren = len(dfn[dfn.val == 1])/len(dfn)
            thresholdn = float(dfn.tail(1)["pred"].values)
        df0 = df[df.pred >= 0.0]
        score0 = len(df0[df0.val == 1])/len(df0)
        threshold0 = 0.0 # float(df0.tail(1)["pred"].values)
        return ((threshold1, threshold2, thresholdn, threshold0),(score1,score2,scoren, score0))

    def cal_accuracy2(self, npFeat, npLabel, thresholds, is_short=False):
        y_pred = self.cls.predict_proba(npFeat)
        if is_short:
            df = pd.DataFrame({"pred": y_pred[:,0], "val": 1-npLabel})
        else:
            df = pd.DataFrame({"pred": y_pred[:,1], "val": npLabel})
        df.sort_values(["pred"], ascending=False, inplace=True)
        df1 = df[df.pred >= thresholds[0]]
        score1 = len(df1[df1.val == 1])/len(df1)
        df2 = df[df.pred >= thresholds[1]]
        score2 = len(df2[df2.val == 1])/len(df2)
        dfn = df[df.pred >= thresholds[2]]
        if len(dfn) == 0:
            scoren = 0.0
        else:
            scoren = len(dfn[dfn.val == 1])/len(dfn)
        df0 = df[df.pred >= 0.0]
        score0 = len(df0[df0.val == 1])/len(df0)
        threshold0 = float(df0.tail(1)["pred"].values)
        return (score1,score2,scoren, score0)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval*5 != 0:
            return
        print("")
        print("LONG...")
        (thresholds, scores) = self.cal_accuracy(self.npFeatTest, self.npLabelTest)
        print("TEST: ", end='')
        for i in range(len(thresholds)):
            print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
        print()
        print("VALD: ", end='')
        scores = self.cal_accuracy2(self.npFeatVal, self.npLabelVal, thresholds)
        for i in range(len(thresholds)):
            print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
        print()
        print()

        is_first = True
        for df_year in self.df_years:
            npFeat, npLabel = base.extract_feat_label(df_year, self.score,drop=True)
            year = df_year['yyyy'].unique()[0]
            print("%s: " % year, end='')
            if is_first:
                (thresholds, scores) = self.cal_accuracy(npFeat, npLabel)
                for i in range(len(thresholds)):
                    print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
                is_first = False
            else:
                scores = self.cal_accuracy2(npFeat, npLabel, thresholds)
                for i in range(len(thresholds)):
                    print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
            print()
        print("SHORT...")
        (thresholds, scores) = self.cal_accuracy(self.npFeatTest, self.npLabelTest, is_short=True)
        print("TEST: ", end='')
        for i in range(len(thresholds)):
            print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
        print()
        print("VALD: ", end='')
        scores = self.cal_accuracy2(self.npFeatVal, self.npLabelVal, thresholds, is_short=True)
        for i in range(len(thresholds)):
            print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
        print()
        print()

        is_first = True
        for df_year in self.df_years:
            npFeat, npLabel = base.extract_feat_label(df_year, self.score,drop=True)
            year = df_year['yyyy'].unique()[0]
            print("%s: " % year, end='')
            if is_first:
                (thresholds, scores) = self.cal_accuracy(npFeat, npLabel, is_short=True)
                for i in range(len(thresholds)):
                    print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
                is_first = False
            else:
                scores = self.cal_accuracy2(npFeat, npLabel, thresholds, is_short=True)
                for i in range(len(thresholds)):
                    print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
            print()
