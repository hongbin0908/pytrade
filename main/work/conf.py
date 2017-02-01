import os
import sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.ta.ta_set import TaSetBase1
from main.base.score2 import ScoreLabel
from main.yeod import yeod
from main.model.spliter import StaticSpliter
from main.classifier.tree import MyRandomForestClassifier

class MltradeConf:
    def __init__(self, window, classifier = MyRandomForestClassifier(),
                 model_split=StaticSpliter(2000,2010,2,1700,2000),
                 valid_split=StaticSpliter(2010,2017,1,1700,2000),
                 score1=ScoreLabel(5, 1.0), score2 = ScoreLabel(5, 1.0),
                 ta = TaSetBase1(), n_pool=10, index="dow30"):
        self.window = window
        self.classifier = classifier
        self.model_split = model_split
        self.valid_split = valid_split
        self.n_pool = n_pool
        self.score1 = score1
        self.score2 = score2
        self.index = index

        self.ta = ta
        self.name = "model_{index}_w{window}_c{classifier}_m{model_split}_v{valid_split}_s{score1}-{score2}_ta{ta}".format (
            **{"index": self.index,
                "window": self.window,
             "classifier":self.classifier.get_name(),
             "model_split": self.model_split.get_name(),
             "valid_split": self.valid_split.get_name(),
             "score1": self.score1.get_name(), "score2":self.score2.get_name(),
             "ta":self.ta.get_name()})

        self.model_split.set_classifier(self.classifier)
        self.valid_split.set_classifier(self.classifier)
        if index == "sp500":
            self.syms = yeod.get_sp500_list(self.window)
        elif index == "dow30":
            self.syms = yeod.get_dow30_list(self.window)
        elif index == "test":
            self.syms = yeod.get_test_list(self.window)
        elif index == "sp100_snapshot_20081201":
            self.syms = yeod.get_sp100_snapshot_20081201(self.window)
        elif index == "sp100_snapshot_20091129":
            self.syms = yeod.get_sp100_snapshot_20091129(self.window)
        elif index == "sp100_snapshot_20100710":
            self.syms = yeod.get_sp100_snapshot_20100710(self.window)
        elif index == "sp100_snapshot_20120316":
            self.syms = yeod.get_sp100_snapshot_20120316(self.window)
        elif index == "sp100_snapshot_20120316":
            self.syms = yeod.get_sp100_snapshot_20120316(self.window)
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
