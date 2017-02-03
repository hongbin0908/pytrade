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
class YearSpliter(BinarySpliter):
    def __init__(self, test_start, test_end, train_start, train_end):
        self.test_start = test_start+ "-01-01"
        self.test_end = test_end+"-01-01"
        self.train_start = train_start+"-01-01"
        self.train_end = train_end+"-01-01"
        
