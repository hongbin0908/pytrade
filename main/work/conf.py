import os
import sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.ta.ta_set import TaSetBase1
from main.base.score2 import ScoreLabel
from main.yeod import yeod
from main.classifier.tree import MyRandomForestClassifier

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

    def get_out_file_prefix(self):
        return os.path.join(root, "report", "%s"
                                 % (self.name))

    def get_dump_name_prefix(self, conf_name):
        return os.path.join(root, 'data',
                                 'model',
                                 self.name + "-" + conf_name + ".model")
    def get_ta_file(self):
        return os.path.join(root, "data", "ta", "%s-%s-%d-%s.pkl"
                            % (self.syms.get_name(), 
                                self.ta.get_name(), 
                                self.week, 
                                self.score1.get_name()))
    def get_pred_file(self):
        return os.path.join(root, "data", "pred", "%s.pred.pkl" % self.name)
