#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import model.model_param_set as param_set


ta =  param_set.d_dir_ta

l_ta = ["ta1"]
l_model = [
        "ta1_GBCv1n1000md3_l1_s2000e2009",
        "ta1_GBCv1n1000md3_l2_s2000e2009",
        "ta1_GBCv1n1000md3_l3_s2000e2009",
        ]

l_label = ["label3"]

l_range = [("2012-01-01", '2015-12-31')]

l_th = [0.7, 0.5]

l_params = []
for th in l_th:
    for model in l_model:
        for label in l_label:
            for drange in l_range:
                for t in l_ta:
                    l_params.append(( model, ta[t], label, drange, th))

