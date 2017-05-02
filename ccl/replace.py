import sys
import os
tmp_str = ""
tmp_index = 0
for j in sys.stdin:
	line_list = j.strip().split(" ")
	if line_list[0] != tmp_str:
		tmp_index += 1
		tmp_str = line_list[0]
		line_list[0] = str(tmp_index)
		print(' '.join(line_list))
	else:
		line_list[0] = str(tmp_index)
		print(' '.join(line_list))
