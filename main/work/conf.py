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
from main.model.spliter import YearSpliter
from main.classifier.tree import RFCv1n2000md6msl100
from main.selector.selector import MiSelector

class MltradeConf:
    def __init__(self, model_split, 
                 classifier=MyRandomForestClassifier(),
                 scores=[ScoreLabel(5, 1.0), ScoreRelative(5), ScoreRelativeOpen(5)],
                 ta=TaSetBase1(), selector=None, n_pool=10, index="dow30", week=0):
        self.model_split = model_split
        self.classifier = classifier
        self.n_pool = n_pool
        self.scores = scores
        self.index = index
        self.week = week
        self.force = False
        self.ta = ta
        #self.relative = False
        if selector is None:
            self.selector = MiSelector([self])
        else:
            self.selector = selector
        

        self.name_ta = "%s_%s" % (self.index, self.ta.get_name())
        self.name_score = ""
        for score in self.scores:
            self.name_score += "%s_" % score.get_name()
        self.name_bitlize = "%s_%s_%s_%s" % (self.name_ta, self.model_split.train_start, self.model_split.train_end, self.scores[0].get_name())

        self.name_sel = "%s" % self.selector.get_name()
        self.name_clazz = "%s_%s" % (self.name_sel, self.classifier.get_name())

        if index == "test":
            self.syms = yeod.get_test_list()
        elif index == "sp100_snapshot_20081201":
            self.syms = yeod.get_sp100_snapshot_20081201()
        elif index == "sp100_snapshot_20091129":
            self.syms = yeod.get_sp100_snapshot_20091129()
        elif index == "sp100_snapshot_20100710":
            self.syms = yeod.get_sp100_snapshot_20100710()
        elif index == "sp100_snapshot_20120316":
            self.syms = yeod.get_sp100_snapshot_20120316()
        elif index == "sp100_snapshot_20140321":
            self.syms = yeod.get_sp100_snapshot_20140321()
        elif index == "sp100_snapshot_20151030":
            self.syms = yeod.get_sp100_snapshot_20151030()
        else:
            self.syms = yeod.sp500_snapshot(index)


    def get_years(self, df):
        if "yyyy" not in df:
            df['yyyy'] = df.date.str.slice(0,4)
        years = df.sort_values(["yyyy"], ascending=True)["yyyy"].unique()
        return years

    def get_classifier_file(self):
        return os.path.join(root, 'data', 'clazz', self.name_clazz)

    
    def get_ta_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'ta')):
            os.makedirs(os.path.join(root,'data','ta'))
        return os.path.join(root, "data", "ta", "%s.pkl" % self.name_ta)

    def get_bitlize_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'bitlize')):
            os.makedirs(os.path.join(root, 'data', 'bitlize'))
        return os.path.join(root, "data", "bitlize", "%s.pkl" % self.name_bitlize)
        
    def get_feat_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'feat')):
            os.makedirs(os.path.join(root, 'data', 'feat'))
        return os.path.join(root, "data", "feat", "%s.pkl" % self.name_bitlize)

    def get_score_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'score')):
            os.makedirs(os.path.join(root, 'data', 'score'))
        return os.path.join(root, 'data', 'score', "%s.pkl" % self.name_score)

    def get_sel_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'sel')):
            os.makedirs(os.path.join(root, 'data', 'sel'))
        return os.path.join(root, 'data', 'sel', "%s.pkl" % self.name_sel)

    def get_pred_file(self):
        if not os.path.exists(os.path.join(root, 'data', 'pred')):
            os.makedirs(os.path.join(root, 'data', 'pred'))
        return os.path.join(root, "data", "pred", "%s.pkl" % self.name_clazz)
class MyConfStableLTa(MltradeConf):
    def __init__(self, ta = ta_set.TaSetBase1Ext4(),
            classifier=RFCv1n2000md6msl100(),
            train_start="1900",
            train_end = "2010",
            index="sp500_snapshot_20091231",
            score=5
            ):

        model_split=YearSpliter(train_end, "2017", train_start, train_end)
        #index="sp100_snapshot_20091129"
        week=-1
        MltradeConf.__init__(self,
                model_split=model_split,
                classifier=classifier,
                ta = ta, n_pool=30, index=index, week = week)

class MyConfStableLTa2(MltradeConf):
    def __init__(self, train_end, test_start):
        model_split=YearSpliter(test_start, "2017", "1900", train_end)
        ta = ta_set.TaSetBase1Ext4El()
        classifier=RFCv1n2000md6msl100()
        index="sp500_snapshot_20091231"
        score=5
        #index="sp100_snapshot_20091129"
        week=-1
        MltradeConf.__init__(self,
                model_split=model_split,
                classifier=classifier,
                ta = ta, n_pool=30, index=index, week = week)
