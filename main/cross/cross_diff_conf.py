#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import model.model_param_set as param_set


ta =  param_set.d_dir_ta

l_ta = [ta["ta1"]]
l_model = [
        "ta1_GBCv1n1000md3_l3_s2000e2009",
        "ta1_GBCv1n1000md3_l3_s2001e2010",
        "ta1_GBCv1n1000md3_l3_s2002e2011",
        ]

l_label = ["label3"]

l_range = [("2012-01-01", '2015-12-31')]

l_th = [0.5, 0.6, 0.7]

l_params = []
for ta in l_ta:
    for model in l_model:
        for label in l_label:
            for drange in l_range:
                for th in l_th:
                    l_params.append(( model, ta, label, drange, th))

