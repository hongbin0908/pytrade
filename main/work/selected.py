#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import pandas as pd


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.work.conf import MyConfStableLTa
def work(confer):
    if not os.path.exists(confer.get_sel_file()) or  confer.force:
        df = confer.selector.work()
        print(df.shape)
        df.to_pickle(confer.get_sel_file())
if  __name__ == "__main__":
    from main.classifier.tree import MySGDClassifier

    confer = MyConfStableLTa(classifier=MySGDClassifier(),score=5)
    ta_file = pd.read_pickle(confer.get_bitlize_file())

    ta_file1 = confer.selector._select(ta_file,
            confer.model_split.train_start,
            confer.model_split.train_end,
            confer.scores[0].get_name())
    ta_file2 = confer.selector._select(ta_file,
            "2013-01-01",
            "2014-01-01",
            confer.scores[0].get_name())
    ta = ta_file2.merge(ta_file1, left_index=True, right_index=True)
    print(ta.head())
