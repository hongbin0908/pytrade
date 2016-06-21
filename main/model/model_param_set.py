#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


d_dir_ta =  {
    "ta1":os.path.join(root, "data", 'ta1'),
    "ta1s1":os.path.join(root, "data", 'ta1s1'),
    "ta1s2":os.path.join(root, "data", 'ta1s2'),
    "ta1s3":os.path.join(root, "data", 'ta1s3'),
    "ta1s4":os.path.join(root, "data", 'ta1s4'),
    "ta1s5":os.path.join(root, "data", 'ta1s5'),
    "taselect":os.path.join(root, "data", 'taselect'),
    "ta2":os.path.join(root, "data", 'ta2'),
    "ta3":os.path.join(root, "data", 'ta_dow'),
    "tatech":os.path.join(root, "data", 'tatech'),
    }
d_model = {
        "DC":DecisionTreeClassifier(),
        "GBCv1n500md3":GradientBoostingClassifier(**{'verbose':1,'n_estimators':500, 'max_depth':3}),
        "GBCv1n1000md3":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3}),
        "GBCv1n1000md3":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3}),
        "GBCv1n1000md9":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':9}),
        "GBCv1n1000md9":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':9}),
        "GBCv1n1000md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n5000md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':5000, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n70md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':70, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n500md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':500, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n400md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':400, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n300md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':300, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n320md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':300, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n2000md3lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':2000, 'max_depth':3, 'learning_rate':0.05}),
        "GBCv1n5000md4lr001":GradientBoostingClassifier(**{'verbose':1,'n_estimators':5000, 'max_depth':4, 'learning_rate':0.05}),
        "GBCv1n1000md3lr02":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3, 'learning_rate':0.2}),
        "GBCv1n1000md3mf05":GradientBoostingClassifier(**{'verbose':1,'n_estimators':1000, 'max_depth':3, 'max_features':0.5}),
        }

d_label = {
        "l3":'label3',
        "l2":'label2',
        "l1":'label1',
        "l4":'label4',
        "l5":'label5',
        'l6':'label6',
        'l8':'label8',
        'l10':'label10',
        'l15':'label15',
        'l20':'label20',
        'l30':'label30',
        'l60':'label60',
        }
d_date_range = {
        "s2000e2009":("2000-01-01", '2009-12-31'),
        "s1700e2009":("1700-01-01", '2009-12-31'),

        "s2001e2010":("2001-01-01", '2010-12-31'),
        "s2002e2011":("2002-01-01", '2011-12-31'),
        "s2003e2012":("2003-01-01", '2012-12-31'),
        "s2004e2013":("2004-01-01", '2013-12-31'),
        "s2005e2014":("2005-01-01", '2014-12-31'),
        "s2006e2015":("2006-01-01", '2015-12-31'),
        "s2010e2015":("2006-01-01", '2015-12-31'),
        "s2014e2015":("2006-01-01", '2015-12-31'),
        "s2005e2009":("2005-01-01", '2009-12-31'),
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
