#!/bin/sh
############################
#@author hongbin@youzan.com
#@date
#@desc TODO
############################
export PATH=/usr/bin:$PATH
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script
cd "$SCRIPT_PATH"/../
 
#./main/ta/build.py  dow call1s1 5
#./main/paper/paper.py  call1s1_dow_GBCv1n322md3lr001_label5_1700-01-01_2009-12-31  146 call1s1_sp500Top50 2010-01-01 2016-12-31 1 0.605
#./main/paper/paper.py  call1s1_dow_GBCv1n322md3lr001_label5_1700-01-01_2009-12-31  146 call1s1_sp500Top50 2010-02-01 2016-12-31 1 0.605
#./main/paper/paper.py  call1s1_dow_GBCv1n400md3lr001_label5_1700-01-01_2009-12-31.se  285 call1s1_dow 2010-01-01 2016-12-31 1 0.60
#./main/paper/paper.py  call1s1_sp500Top50_GBCv1n500md3lr001_label5_1700-01-01_2009-12-31  300 call1s1_sp500Top50 2010-01-01 2016-12-31 1 0.605
./main/paper/paper.py  call1s1_sp500Top10_GBCv1n500md3lr001_label5_1700-01-01_2009-12-31  300 call1s1_sp500Top10 2010-01-01 2016-12-31 1 0.62

