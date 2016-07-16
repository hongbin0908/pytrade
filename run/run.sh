#!/bin/sh
############################
#@author hongbin@youzan.com
#@date
#@desc TODO
############################
export PATH=/usr/bin:$PATH
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script
pushd . > /dev/null
cd "$SCRIPT_PATH"/../

last_date=`python -c "import main.base as base; print base.last_trade_date()"`
ta=call1s1
eod=sp500Top100
batch=50
model=GBCv1n1000md3lr001-${ta}-sp500Top100-${batch}-label5-1700-01-01-2009-12-31-0-0
 
rm -rf ./data/yeod_batch/${eod}-${batch}
rm -rf ./data/yeod/index_dow
./main/yeod/yeod.py index_dow  1
./main/yeod/yeod_b.py ${eod} ${batch} 1
./main/ta/build_b.py  ${eod} ${batch} ${ta} 1
./main/paper/paper_b.py ${model} 650 ${ta}-${eod} ${batch} 2010-06-01 2016-06-31  2  400
./main/pred/pred_b.py  ${model}  600  ${ta}-${eod}-${batch} ${last_date} ${last_date} label5
popd  > /dev/null # return the directory orignal
