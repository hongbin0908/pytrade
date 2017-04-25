#!/bin/sh
#########################
#@author hongbin0908@126.com
#@date
#@desc TODO
#########################
export PATH=/usr/bin:$PATH
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script
pushd . > /dev/null
cd "$SCRIPT_PATH"

python3 ${SCRIPT_PATH}/get_yeod.py
rsync -avH /data/users/hongbin/pytrade hongbin@bd-r1hdp22:~/
ssh hongbin@bd-r1hdp22 "/usr/bin/python3 /data/users/hongbin/pytrade/run/run.py  -f"

popd   > /dev/null # return the directory orignal
