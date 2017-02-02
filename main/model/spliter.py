import sys
import os
import numpy as np


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)


#class Token:
#    def __init__(self, classifier, train, test):
#        self.classifier = classifier
#        self.train = train
#        self.test = test
class Token:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class BaseSpliter:
    def split(self):
        assert False

class BinarySpliter(BaseSpliter):
    def __init__(self, test_start, test_end, train_start, train_end):
        self.test_start = test_start
        self.test_end = test_end
        self.train_start = train_start
        self.train_end = train_end
    def split(self, df):
        df['yyyy'] = df.date.str.slice(0,4)
        df_train = df[(df.date >= self.train_start) & (df.date < self.train_end)]
        df_test =  df[(df.date >= self.test_start)  & (df.date < self.test_end)]
        df_train.name = "train_%s_%s" % (self.train_start, self.train_end)
        df_test.name = "test_%s_%s" % (self.test_start, self.test_end)
        return Token(df_train, df_test)
    def get_name(self):
       return "%s-%s-%s-%s" % \
              (self.train_start, self.train_end,
               self.test_start, self.test_end)

class StaticSpliter(BaseSpliter):
    def __init__(self, start, end, len, model_start, model_end):
        self.start = start
        self.end = end
        self.len = len
        self.model_start = model_start
        self.model_end = model_end
    def set_classifier(self, classifier):
        if classifier is not None:
            self.classifier = classifier

    def _split_years(self):
        cur_idx = self.start
        while True:
            cur_win = []
            for i in range(cur_idx, cur_idx + self.len):
                cur_win.append(str(i))
            yield cur_win
            cur_idx += self.len
            if cur_idx + self.len > self.end:
                break

    def split(self, df):
        df = df.replace([np.inf,-np.inf],np.nan).dropna()
        df['yyyy'] = df.date.str.slice(0,4)
        df_model = df[ (df.date.str.slice(0,4) >= str(self.model_start)) & (df.date.str.slice(0,4) <str(self.model_end))]
        df_model.name = "%d_%d" % (self.model_start, self.model_end)
        for each in self._split_years():
            yield Token(self.classifier, df_model, df[(df.date.str.slice(0,4) >= each[0]) & (df.date.str.slice(0,4) <= each[-1])])

    def get_name(self):
       return "%d-%d-%d-%d-%d" % \
              (self.start, self.end, self.len,
               self.model_start,
               self.model_end)
