import sys
import os
import numpy
import pandas

def load_data(filename):
	try:
		info = pandas.read_csv(filename, delimiter=',', header=1)
	except Exception as e:
		sys.stderr.write("error: info = %s\n" %(e))
		sys.exit(1)
	return info

def build_sample(info, stock_name = "A", interval = 15):
	for i in range(interval, info.shape[0]-1):
		x_date = info.iloc[i-interval+1, 0]
		x = info.iloc[i-interval+1:i+1, 5].tolist()
		x_str = ' '.join(map(str, x))
		label = info.iloc[i+1, 8] * 1.0 / info.iloc[i, 8]

		#if label > 1:
		#	label = "1"
		#else:
		#	label = "0" 
		#label_str = ["0", "0", "0"]
		#if label >1.005:
		#	label_str[0] = "1"
		#elif label < 0.995:
		#	label_str[2]  = "1"
		#else:
		#	label_str[1] = "1"
		print("%s %s %s %.4f" %(stock_name, x_date, x_str, label))
if __name__ == "__main__":
	fd = open("stocks_part")
	for i in fd:
		result = load_data(i.strip())
		stock_name = i.strip().split("/")[-1].replace(".csv", "")
		build_sample(result, stock_name, 15)
