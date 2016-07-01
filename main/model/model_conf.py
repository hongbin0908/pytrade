#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


l_params = [
    #"ta1s1_GBCv1n1000md3lr001_l5_s2000e2009",
    #"tadowcall1_GBCv1n1000md3lr001_l5_s2000e2009",
    #"tadowcall1_GBCv1n1000md3lr001_l10_s2000e2009",

    #"tadowcall1_GBCv1n1000md3lr001_l5_s1700e2009",
    "tadowcall1_GBCv1n322md3lr001_l5_s1700e2009",

    "call1s1_dow_GBCv1n322md3lr001_l5_s1700e2009",
    "call1s1_sp500_GBCv1n1000md3lr001_l5_s1700e2009",


        #"ta1s1_GBCv1n1000md3lr001_l5_s2000e2009",
        #"ta1s1_GBCv1n1000md3lr001_l5_s2006e2015",

        #"taselect_GBCv1n1000md3lr001_l5_s2006e2015",
        #"taselect_GBCv1n1000md3lr001_l5_s2005e2014",
        #"taselect_GBCv1n1000md3lr001_l5_s2004e2013",
        #"taselect_GBCv1n1000md3lr001_l5_s2003e2012",
        #"taselect_GBCv1n1000md3lr001_l5_s2002e2011",
        #"taselect_GBCv1n1000md3lr001_l5_s2001e2010",
        #"taselect_GBCv1n1000md3lr001_l5_s2000e2009",
        #
        #"taselect_GBCv1n2000md3lr001_l5_s2006e2015",
        #"taselect_GBCv1n2000md3lr001_l5_s2005e2014",
        #"taselect_GBCv1n2000md3lr001_l5_s2004e2013",
        #"taselect_GBCv1n2000md3lr001_l5_s2003e2012",
        #"taselect_GBCv1n2000md3lr001_l5_s2002e2011",
        #"taselect_GBCv1n2000md3lr001_l5_s2001e2010",
        #"taselect_GBCv1n2000md3lr001_l5_s2000e2009",
        #
        #"taselect_GBCv1n1000md3lr001_l5_s2006e2015",
        #"taselect_GBCv1n2000md3lr001_l5_s2006e2015",
        #"taselect_GBCv1n1000md3lr001_l5_s2000e2009",
        #"taselect_GBCv1n500md3lr001_l5_s2000e2009",
        #"taselect_GBCv1n2000md3lr001_l5_s2000e2009",
        #
        #"ta1_DC_l5_s2000e2009",
        #
        #"ta1_GBCv1n1000md3lr001_l5_s1700e2009",
        #"ta1_GBCv1n1000md3lr001_l5_s2000e2009",
        #"ta1_GBCv1n70md3lr001_l5_s2000e2009",
        #"ta1_GBCv1n400md3lr001_l5_s2000e2009",
        #"ta1s2_GBCv1n400md3lr001_l5_s2000e2009",
        #"ta1s3_GBCv1n400md3lr001_l5_s2000e2009",
        #"ta1s3_GBCv1n300md3lr001_l5_s2000e2009",
        #
        #"tadow_GBCv1n400md3lr001_l5_s2000e2009",
        #"tadow_GBCv1n400md3lr001_l10_s2000e2009",
        #"tadow_GBCv1n400md3lr001_l10_s2000e2009",
        #"tadow_GBCv1n400md3lr001_l5_s1700e2009",
        #
        #
        #"ta1s4_GBCv1n500md3lr001_l5_s2000e2009",
        #"ta1s4_GBCv1n320md3lr001_l5_s2000e2009",
        #"ta1s4_GBCv1n320md3lr001_l5_s2001e2010",
        #"ta1s4_GBCv1n320md3lr001_l5_s2002e2011",
        #"ta1s4_GBCv1n320md3lr001_l5_s2003e2012",
        #"ta1s4_GBCv1n320md3lr001_l5_s2004e2013",
        #"ta1s4_GBCv1n320md3lr001_l5_s2005e2014",
        #"ta1s4_GBCv1n320md3lr001_l5_s2006e2015",
        #"ta1s5_GBCv1n500md3lr001_l5_s2000e2009",
        #"ta1s4_GBCv1n500md3lr001_l5_s2000e2009",
        #"ta1s5_GBCv1n320md3lr001_l5_s2000e2009",
        #"ta1s5_GBCv1n320md3lr001_l5_s2001e2010",
        #"ta1s5_GBCv1n320md3lr001_l5_s2002e2011",
        #"ta1s5_GBCv1n320md3lr001_l5_s2003e2012",
        #"ta1s5_GBCv1n320md3lr001_l5_s2004e2013",
        #"ta1s5_GBCv1n320md3lr001_l5_s2005e2014",
        #"ta1s5_GBCv1n320md3lr001_l5_s2006e2015",
        #"ta1_GBCv1n1000md3lr001_l5_s2001e2010",
        #"ta1_GBCv1n1000md3lr001_l5_s2002e2011",
        #"ta1_GBCv1n1000md3lr001_l5_s2003e2012",
        #"ta1_GBCv1n1000md3lr001_l5_s2004e2013",
        #"ta1_GBCv1n1000md3lr001_l5_s2005e2014",
        #"ta1_GBCv1n1000md3lr001_l5_s2006e2015",
        #"ta1_GBCv1n1000md3lr001_l5_s2010e2015",
        #"ta1_GBCv1n1000md3lr001_l5_s2014e2015",
        #"ta1_GBCv1n5000md3lr001_l5_s2000e2009",
        #"ta1_GBCv1n5000md4lr001_l5_s2000e2009",
        #"ta1_GBCv1n1000md3lr02_l5_s2000e2009",
        #"ta1_GBCv1n1000md3mf05_l5_s2000e2009",
        #
        #"tatech_GBCv1n1000md3_l5_s2000e2009",
        #
        #"ta1_GBCv1n500md3_l3_s2005e2009",
        #"ta1_GBCv1n500md3_l3_s2000e2009",
        #"ta1_GBCv1n1000md3_l3_s2000e2009",
        #"ta1_GBCv1n1000md3_l3_s2001e2010",
        #"ta1_GBCv1n1000md3_l3_s2002e2011",
        #"ta1_GBCv1n1000md3_l3_s2003e2012",
        #"ta1_GBCv1n1000md3_l3_s2004e2013",
        #"ta1_GBCv1n1000md3_l3_s2005e2014",
        #
        #"ta1_GBCv1n1000md3_l3_s2005e2009",
        #
        #"ta1_GBCv1n1000md3_l2_s2000e2009",
        #"ta1_GBCv1n1000md3_l1_s2000e2009",
        #"ta1_GBCv1n1000md3_l4_s2000e2009",
        #"ta1_GBCv1n1000md3_l5_s2000e2009",
        #"ta1_GBCv1n1000md3_l6_s2000e2009",
        #"ta1_GBCv1n1000md3_l8_s2000e2009",
        #"ta1_GBCv1n1000md3_l10_s2000e2009",
        #"ta1_GBCv1n1000md3_l15_s2000e2009",
        #"ta1_GBCv1n1000md3_l20_s2000e2009",
        #"ta1_GBCv1n1000md3_l30_s2000e2009",
        #"ta1_GBCv1n1000md3_l60_s2000e2009",
        #
        #
        #"ta3_GBCv1n1000md3_l5_s2000e2009",
        #"ta3_GBCv1n1000md3_l3_s2000e2009",
        #
        #"ta2_GBCv1n1000md3_l3_s2000e2009",

        ]
