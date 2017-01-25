#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import model.model_param_set as param_set

"""
## 研究目的
    研究模型在时间上的稳定性
## 结论
"""
ta =  param_set.d_dir_ta


l_params = [
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.0),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.5),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.6),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.65),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.7),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.75),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.8),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-12-31'),  0.0),

        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.5),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.6),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.65),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.7),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.75),
        #("ta1_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1"], "label5", ("2016-01-01", '2016-03-31'),  0.8),

        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.5),
        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.6),
        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.65),
        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.7),
        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.75),
        ("ta1s4_GBCv1n1000md3lr001_l5_s2006e2015" , ta["ta1s4"], "label5", ("2016-03-01", '2016-06-31'),  0.8),
        ]

