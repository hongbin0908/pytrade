import os,sys
import pandas as pd
import numpy as np


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.base.timer import Timer
from main.work import build


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(root, "./data/ta/sp100w150i0-TaBase1Ext4El-score_label_5_100.pkl"))
    # df = pd.read_pickle(os.path.join(root, "./data/ta/Dow30w2i0-base1-score_label_5_100.pkl"))
    for feat_name in ["ta_RSI_7_d1_0", "ta_RSI_7_d2_0", "ta_NATR_7_d2_3", "ta_ROC_10_d2_0", "ta_ROCR100_10_d2_0", "ta_CMO_14_d2_1",
                      "ta_ROCR100_10_d1_0", "ta_ROC_10_d1_0", "ta_ROCR100_7_d2_0", "ta_ROC_7_d2_0"]:
        df1 = df[df[feat_name] == 1]
        print("%s:%f" % (feat_name, len(df1[df1.score_label_5_100 == 1])/len(df1)))

    for feat_name in ["ta_CMO_14_d2_0", "ta_ROCP_10_d1_0", "ta_ROCR_10_d1_0", "ta_ROCR_7_d2_0", "ta_ROCP_7_d2_0", "ta_RSI_14_d2_1",
                      "ta_ROCP_10_d2_0", "ta_ROCR_10_d2_0", "ta_CMO_7_d1_0", "ta_CMO_7_d2_0"]:
        df1 = df[df[feat_name] == 1]
        print("%s:%f" % (feat_name, len(df1[df1.score_label_5_100 == 1])/len(df1)))




