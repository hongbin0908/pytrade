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
d_date_range = {
        "s2000e2009":("2000-01-01", '2009-12-31'),
        "s2001e2010":("2001-01-01", '2010-12-31'),
        "s2002e2011":("2002-01-01", '2011-12-31'),
        "s2003e2012":("2003-01-01", '2012-12-31'),
        "s2004e2013":("2004-01-01", '2013-12-31'),
        "s2005e2014":("2005-01-01", '2014-12-31'),
        }
d_all = {}
for dir_ta in d_dir_ta:
    for model in d_model:
        for label in d_label:
            for date_range in d_date_range:
                name = "%s_%s_%s_%s" % (dir_ta, model, label, date_range)
                d_all[name] = (
                    d_dir_ta[dir_ta], \
                    d_model[model], \
                    d_label[label], \
                    d_date_range[date_range]
                    )
