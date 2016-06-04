#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
from sklearn.ensemble import GradientBoostingClassifier
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

d_dir_ta =  {"ta1":os.path.join(root, "data", 'ta1')}
d_model = {
        "GBCv1n500md3":GradientBoostingClassifier(**{'verbose':1,'n_estimators':500, 'max_depth':3}),
        "GBCv1n1000md3":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3}),
        }

d_label = {
        "l3":'label3',
        "l2":'label2',
        "l1":'label1',
        }
d_date_range = {"s2000e2009":("2000-01-01", '2009-12-31')}
