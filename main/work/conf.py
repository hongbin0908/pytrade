import os
import sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.ta.ta_set import TaSetBase1
from main.ta import ta_set
from main.score.score import ScoreLabel
from main.score.score import ScoreRelative
from main.score.score import ScoreRelativeOpen
from main.yeod import yeod
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import MyLogisticRegressClassifier
from main.model.spliter import YearSpliter
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import ccl2
from main.classifier.tree import cnn
from main.classifier.tree import MyMdnClassifier
from main.classifier.ts import Ts
from main.classifier.logit2 import Logit2
from main.classifier.tf_dnn import TfDnn
from main.selector.selector import MiSelector
from main import base

class MltradeConf:
    def __init__(self, model_split, 
                 classifier=MyRandomForestClassifier(),
                 scores=[ScoreLabel(5, 1.0), ScoreRelative(5), ScoreRelativeOpen(5)],
                 ta=TaSetBase1(),  is_adj = True, selector=None, n_pool=10,
                 syms=yeod.sp500_snapshot("sp500_snapshot_20091231"),
                 week=0, model_postfix="", train_iters = 1):
        self.model_split = model_split
        #self.classifier = classifier
        self.n_pool = n_pool
        self.scores = scores
        self.syms = syms
        self.week = week
        self.force = False
        self.ta = ta
        self.is_adj = is_adj
        self.last_trade_date = base.get_last_trade_date_local(self.syms.get_name())
        self.model_postfix = model_postfix
        self.train_iters = train_iters
        if selector is None:
            self.selector = MiSelector([self])
        else:
            self.selector = selector


        self.name_bitlize = "%s_%s_%s_%s" % (self.name_ta(), self.model_split.train_start,
                                                self.model_split.train_end, self.scores[0].get_name())
        self.name_sel = "%s" % (self.selector.get_name())

    def get_name_score(self):
        name_score = ""
        for score in self.scores:
            name_score += "%s_" % score.get_name()
        name_score += self.last_trade_date
        return name_score

    def get_name_clazz(self):
        return "%s_%s_%d_%s_%s_%s_%s" % (self.syms.get_name(), self.ta.get_name(), self.is_adj,
                                                  self.get_name_score(), self.model_split.train_start, self.model_split.train_end,
                                                  self.classifier.get_name())
    def name_ta(self):
        return "%s_%s_%d_%s_%s" % (self.syms.get_name(), self.ta.get_name(), self.is_adj, self.get_name_score(), self.last_trade_date)
    def get_years(self, df):
        if "yyyy" not in df:
            df['yyyy'] = df.date.str.slice(0,4)
        years = df.sort_values(["yyyy"], ascending=True)["yyyy"].unique()
        return years

    def get_yeod_dir(self):
        return os.path.join(root, 'data', 'yeod', self.syms.get_name() + '_' + self.last_trade_date)

    def get_classifier_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'clazz')):
            os.makedirs(os.path.join(root, 'data', 'clazz'))
        return os.path.join(root, 'data', 'clazz', self.get_name_clazz() + "_" +self.model_postfix)

    
    def get_ta_file(self):
        name_ta = "%s_%s_%d_%s" % (self.syms.get_name(), self.ta.get_name(), self.is_adj, self.last_trade_date)
        if not os.path.exists(os.path.join(root, 'data', 'ta')):
            os.makedirs(os.path.join(root,'data','ta'))
        return os.path.join(root, "data", "ta", "%s.pkl" % (name_ta))

    def get_bitlize_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'bitlize')):
            os.makedirs(os.path.join(root, 'data', 'bitlize'))
        return os.path.join(root, "data", "bitlize", "%s.pkl" % (self.name_ta()))
        
    def get_feat_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'feat')):
            os.makedirs(os.path.join(root, 'data', 'feat'))
        return os.path.join(root, "data", "feat", "%s.pkl" % (self.name_ta()))

    def get_score_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'score')):
            os.makedirs(os.path.join(root, 'data', 'score'))
        return os.path.join(root, 'data', 'score', "%s_%s.pkl" % (self.get_name_score() , self.last_trade_date))

    def get_sel_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'sel')):
            os.makedirs(os.path.join(root, 'data', 'sel'))
        return os.path.join(root, 'data', 'sel', "%s.pkl" % (self.name_ta()))

    def get_pred_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'pred')):
            os.makedirs(os.path.join(root, 'data', 'pred'))
        return os.path.join(root, "data", "pred", "%s_%s_%s.pkl" % (self.name_ta(), self.classifier.get_name(), self.model_postfix))

    def get_long_report_file(self):
        return os.path.join(local_path, '..','..',"data", 'report', self.last_trade_date + "_long.txt")

    def get_short_report_file(self):
        return os.path.join(local_path, '..','..',"data", 'report', self.last_trade_date + "_short.txt")

class MyConfStableLTa(MltradeConf):
    def __init__(self, ta = ta_set.TaSetBase1Ext4(), is_adj = False,
            classifier=RFCv1n2000md6msl100(),
            #train_start="1990",
            train_start="2000",
            train_end = "2010",
            syms=yeod.sp500_snapshot("sp500_snapshot_20091231"),
            score=5
            ):

        model_split=YearSpliter(train_end, "2017", train_start, train_end)
        week=-1
        MltradeConf.__init__(self,
                model_split=model_split,
                classifier=classifier,
                scores = [ScoreLabel(score, 1.0), ScoreRelative(score), ScoreRelativeOpen(score)],
                ta = ta, is_adj = is_adj, n_pool=30, syms=syms, week = week)

class MyConfForTest(MltradeConf):
    def __init__(self):
        classifier = cnn(batch_size=10000, nb_epoch=1, verbose=0)
        classifier = Ts(max_iterations=2000)
        classifier = ccl2()
        classifier = Logit2(nb_epoch=1)
        classifier = TfDnn(nb_epoch=1)
        model_split=YearSpliter('2010', "2017", "1990", "2010")
        MltradeConf.__init__(self, model_split=model_split, classifier=classifier, n_pool=1, syms=yeod.SymsForTest())


class MyMdnConfForTest(MltradeConf):
    def __init__(self, inputsiz, hidden_size, model_size, lr, train_Begin, train_end, test_begin, test_end):
        classifier = MyMdnClassifier(inputsiz, hidden_size, model_size, lr)
        model_split = YearSpliter(test_begin, test_end, train_Begin, train_end)
        MltradeConf.__init__(self, model_split=model_split, classifier=classifier, n_pool=3, syms=yeod.SymsForTest())
