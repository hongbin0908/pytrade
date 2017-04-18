import sys
import os
import datetime

b_date = '2017-01-01'
for i in range(0, 31):
	tmp_date = datetime.datetime.strptime(b_date, "%Y-%m-%d") + datetime.timedelta(days=i)
	tmp_date_str = datetime.datetime.strftime(tmp_date, "%Y-%m-%d")
	cmd = "python3 process_rnn_with_select.py " + tmp_date_str
	os.system(cmd)
