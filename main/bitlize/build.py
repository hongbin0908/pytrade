#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author Bin Hong
import os,sys
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.bitlize import bitlize

def work(confer):
    if not os.path.exists(confer.get_bitlize_file()) or  confer.force:
        df_ta = pd.read_pickle(confer.get_ta_file())
        df_ta["sd"] = df_ta["sym"] + df_ta["date"]
        df_ta = df_ta.set_index("sd")

        df_score = pd.read_pickle(confer.get_score_file())
        df_score["sd"] = df_score["sym"] + df_score["date"]
        df_score = df_score.set_index("sd")

        df = pd.concat([df_ta, df_score[[score.get_name() for score in confer.scores]]], axis=1, join_axes=[df_ta.index])

        assert len(df) == len(df_ta)
        df = df.reset_index(drop=True)
        if base.is_test_flag():
            df, df_feat = bitlize.feat_split(df, confer.model_split.train_start, confer.model_split.train_end, 0.5, confer.scores[0].get_name(), 2, 2, confer.n_pool)
        else:
            df, df_feat = bitlize.feat_split(df, confer.model_split.train_start, confer.model_split.train_end, 0.5, confer.scores[0].get_name(), 2, 20000, confer.n_pool)
        df.reset_index(drop=True).to_pickle(confer.get_bitlize_file())
        df_feat.reset_index(drop=True).to_pickle(confer.get_feat_file())
