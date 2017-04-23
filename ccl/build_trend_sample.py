import sys
import os
import numpy

def get_gaussian_sample(avg, va, interval):
	result = numpy.random.normal(avg, va, interval)
	return result

def get_trend_sample(x_point, x_position, a0, a1, interval):
	b = x_point - x_position * a0
	result = []
	for i in range(0, x_position):
		x = a0 * i + b
		result.append(x)
	b = x_point - x_position * a1
	for i in range(x_position, interval):
		x = a1 * i + b
		result.append(x)
	return numpy.array(result)

def build_trend_sample(interval, number):
	for i in range(0, number):
		avg = numpy.random.uniform(0,0.2)
		va = numpy.random.uniform(0, 2)
		result = get_gaussian_sample(avg, va, interval)
		x_point = numpy.random.normal(50, 5)
		x_position = int(numpy.random.uniform(0, interval))
		a0 = numpy.random.normal(0, 1.2)
		a1 = numpy.random.normal(0, 1.2)
		a0 = numpy.fabs(a0)
		a1 = numpy.fabs(a1)
		sign0 = numpy.random.uniform(0,1)
		if sign0 > 0.5:
			sign0 = 1
		else:
			sign0 = -1
		sign1 = numpy.random.uniform(0, 1)
		if sign1 > 0.5:
			sign1 = 1
		else:
			sign1 = -1
		result_1 = get_trend_sample(x_point, x_position, a0*sign0, a1*sign1, interval)
		result_all = result_1 + result
		result_all = result_all/result_all[0]
		result_str = result_all.astype(str)
		result_label = ['0', '0', '0', '0']
		if sign0 * sign1 >0 and sign0>0:
			result_label[0] = '1'
		if sign0 * sign1 > 0 and sign0<0:
			result_label[1] = '1'
		if sign0 * sign1 < 0 and sign0 > 0:
			result_label[2] = '1'
		if sign0 * sign1 < 0 and sign0 < 0:
			result_label[3] = '1'
		print("%s %s" %(' '.join(result_str), ' '.join(result_label) ))

if __name__ == "__main__":
	build_trend_sample(15, 100000)
		
			
