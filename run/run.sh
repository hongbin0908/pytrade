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

yestday=$1
 
rm -rf ./data/yeod_batch/sp500top100
./main/yeod/yeod_b.py sp500Top100 50 10
./main/ta/build_b.py sp500Top100 50 call1s3 10
./main/pred/pred_b.py 
popd  > /dev/null # return the directory orignal
