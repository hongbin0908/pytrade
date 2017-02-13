import os
import sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.ta.ta_set import TaSetBase1
from main.ta import ta_set
from main.base.score2 import ScoreLabel
from main.yeod import yeod
from main.classifier.tree import MyRandomForestClassifier
from main.model.spliter import YearSpliter
from main.classifier.tree import RFCv1n2000md6msl100

class MltradeConf:
    def __init__(self, model_split,
                 classifier = MyRandomForestClassifier(),
                 score1=ScoreLabel(5, 1.0), score2 = ScoreLabel(5, 1.0),
                 ta = TaSetBase1(), n_pool=10, index="dow30", week=0):
        self.model_split = model_split
        self.classifier = classifier
        self.n_pool = n_pool
        self.score1 = score1
        self.score2 = score2
        self.index = index
        self.week = week

        self.ta = ta
        self.name = "model_{index}_c{classifier}_m{model_split}_s{score1}-{score2}_ta{ta}_week{week}".format (
            **{"index": self.index,
             "classifier":self.classifier.get_name(),
             "model_split": self.model_split.get_name(),
             "score1": self.score1.get_name(), "score2":self.score2.get_name(),
             "ta":self.ta.get_name(),
             "week":self.week})

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
        elif index == "sp500_snapshot_20091231":
            self.syms = yeod.get_sp500_snapshot_20091231()
        else:
            assert(False)


    def get_years(self, df):
        if "yyyy" not in df:
            df['yyyy'] = df.date.str.slice(0,4)
        years = df.sort_values(["yyyy"], ascending=True)["yyyy"].unique()
        return years

    def get_cross_file(self):
        return os.path.join(root, 'data', 'cross', '%s-%s'
            % (self.name, self.syms.get_name()))

    def get_classifier_file(self):
        return os.path.join(root, 'data', 'clazz', self.classifier.get_name() + "-" 
                + self.index + "-"
                + self.model_split.train_start + "-"
                + self.model_split.train_end + "-"
                + self.score1.get_name() + '-'
                + self.ta.get_name() + '-'
                + str(self.week))
    def get_out_file_prefix(self):
        return os.path.join(root, "report", "%s"
                                 % (self.name))

    def get_dump_name_prefix(self, conf_name):
        return os.path.join(root, 'data',
                                 'model',
                                 self.name + "-" + conf_name + ".model")
    def get_ta_bit_file(self):
        return os.path.join(root, "data", "bit", self.syms.get_name() + "-"
                + self.ta.get_name()+".pkl"
                )
    def get_origin_ta_file(self):
        return os.path.join(root, "data", "tao", "%s-%s-%s.pkl"
                            % (self.syms.get_name(), 
                                self.ta.get_name(), 
                                self.score1.get_name()))

    def get_ta_file(self):
        return os.path.join(root, "data", "ta", "%s-%s-%d-%s.pkl"
                            % (self.syms.get_name(), 
                                self.ta.get_name(), 
                                self.week, 
                                self.score1.get_name()))
    def get_pred_file(self):
        return os.path.join(root, "data", "pred", "%s.pred.pkl" % self.name)
class MyConfStableLTa(MltradeConf):
    def __init__(self, ta = ta_set.TaSetBase1Ext4El(),classifier=RFCv1n2000md6msl100()):

        model_split=YearSpliter("2010", "2017", "1900", "2010")
        classifier = classifier
        score=5
        index="sp100_snapshot_20091129"
        week=-1
        MltradeConf.__init__(self,
                model_split=model_split,
                classifier=classifier,
                score1=ScoreLabel(score, 1.0),
                score2 = ScoreLabel(score, 1.0),
                ta = ta, n_pool=30, index=index, week = week)
