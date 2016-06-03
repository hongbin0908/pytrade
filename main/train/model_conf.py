#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author Bin Hong
import sys,os
from sklearn.ensemble import GradientBoostingClassifier
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
import models

d_conf = {
    "conf_20160522": [
                    "CF2016052201",
                    "CF2016052202",
                    "CF2016052203",
                    "CF2016052207",
                    "CF2016052208",
                    "CF2016052209",
                    "CF2016052302",
                    "CF2016052303",
                    "CF2016052304"],
    "conf_20160526": [
                    "CF2016052201",
                    "CF2016052601"],
    "conf_20160527": [
                    "CF2016052201",
                    "CF2016052701"],
    "conf_20160529": [
                    "CF2016052201",
                    "CF2016052901",
                     ],
    "conf_2016052902": [
                    "CF2016052902",
                    "CF2016052901",
                    "CF2016052903",
                     ],
    "conf_20160530": [
                    #"CF2016052901",
                    #"CF2016053001",
                    #"CF2016053101",
                     ],
    "conf_sp500_50_3": ["CF_SP500_50_3"],
    "conf_diff_label": ["CF2016060201", "CF2016060202", "CF2016060203"],
    "conf_select_params": [
            "CF2016060201",
            "CF2016060301",
            "CF2016060302",
            "CF2016060303",
            "CF2016060304",
            "CF2016060305",
            "CF2016060306",
        ],
    }
