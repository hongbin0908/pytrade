#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author Bin Hong
import sys,os
from sklearn.ensemble import GradientBoostingClassifier
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

d_params = {
    "M20160522GBC01": {'verbose':1,'n_estimators':500, 'max_depth':3},
    "M20160522GBC02": {'verbose':0,'n_estimators':50, 'max_depth':5},
    "M20160522GBC03": {'verbose':0,'n_estimators':500, 'max_depth':4},
    "M20160522GBC04": {'verbose':0,'n_estimators':500, 'max_depth':4, 'learning_rate':0.01},
    "M20160522GBC05": {'verbose':0,'n_estimators':500, 'max_depth':4, 'learning_rate':0.2},
    "M20160522GBC06": {'verbose':0,'n_estimators':50, 'max_depth':4},
    "M20160522GBC07": {'verbose':1,'n_estimators':50, 'max_depth':3},
    "M20160529GBC01": {'verbose':1,'n_estimators':100, 'max_depth':3},
}

d_label = {
    "L20160522CL01":1,
    "L20160522CL02":2,
    "L20160522CL03":3,
    "L20160522CL04":4,
    "L20160522CL05":5,
    "L20160522CL06":10,
    "L20160522CL07":30,
    "L20160522CL08":8,
    "L20160522CL09":15,
    "L20160522CL10":20,
}

d_ta = {
    "T20160522TA01": os.path.join(root, 'data','ta1'),
    "T20160522CD01": os.path.join(root, 'data','ta2'),
    "T20160527TA01": os.path.join(root, 'data','ta3'),
    "T20160529TA01": os.path.join(root, 'data','ta4'),
}

d_range = {
        "RG2016052601":(
                    ("1970-01-01", "2016-01-01"),
                    [
                        ('1970-01-01','2010-01-01','2010-01-01', '2016-01-1'),
                    ]
                    ),
        "RG2016052602":(
                    ("1970-01-01", "2016-01-01"),
                    [
                        ('1970-01-01','2015-01-01','2015-01-01', '2016-01-1'),
                        ('1970-01-01','2014-01-01','2014-01-01', '2015-01-1'),
                        ('1970-01-01','2013-01-01','2013-01-01', '2014-01-1'),
                        ('1970-01-01','2012-01-01','2012-01-01', '2013-01-1'),
                        ('1970-01-01','2011-01-01','2011-01-01', '2012-01-1'),
                        ('1970-01-01','2010-01-01','2010-01-01', '2011-01-1'),
                    ]
                    ),
}

d_choris = {
    "CF2016052201": (d_ta["T20160522TA01"], d_label["L20160522CL03"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052202": (d_ta["T20160522TA01"], d_label["L20160522CL02"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052203": (d_ta["T20160522TA01"], d_label["L20160522CL04"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052204": (d_ta["T20160522TA01"], d_label["L20160522CL03"], d_params["M20160522GBC03"], d_range["RG2016052601"]),
    "CF2016052205": (d_ta["T20160522TA01"], d_label["L20160522CL03"], d_params["M20160522GBC04"], d_range["RG2016052601"]),
    "CF2016052206": (d_ta["T20160522CD01"], d_label["L20160522CL03"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052207": (d_ta["T20160522TA01"], d_label["L20160522CL05"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052208": (d_ta["T20160522TA01"], d_label["L20160522CL06"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052209": (d_ta["T20160522TA01"], d_label["L20160522CL07"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052302": (d_ta["T20160522TA01"], d_label["L20160522CL08"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052303": (d_ta["T20160522TA01"], d_label["L20160522CL09"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052304": (d_ta["T20160522TA01"], d_label["L20160522CL10"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052601": (d_ta["T20160522TA01"], d_label["L20160522CL03"], d_params["M20160522GBC01"], d_range["RG2016052602"]),
    "CF2016052701": (d_ta["T20160527TA01"], d_label["L20160522CL03"], d_params["M20160522GBC01"], d_range["RG2016052601"]),
    "CF2016052901": (d_ta["T20160527TA01"], d_label["L20160522CL03"], d_params["M20160522GBC07"], d_range["RG2016052601"]),
    "CF2016052902": (d_ta["T20160529TA01"], d_label["L20160522CL03"], d_params["M20160522GBC07"], d_range["RG2016052601"]),
    "CF2016052903": (d_ta["T20160529TA01"], d_label["L20160522CL03"], d_params["M20160529GBC01"], d_range["RG2016052601"]),
}
