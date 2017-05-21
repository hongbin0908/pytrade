#abtest功能设计
##目标
用于进行模型之间的选择. 目前仅支持针对report脚本中的top10000数据, 进行accuracy统计
##输入方式
执行run.py中的主要逻辑, 多次训练模型并给出结果
##输出内容
生成一个abtest报告, 报告格式为:
[model1]
top10000[round N]: accuracy
[model2]
top10000[round N]: accuracy
[summary]
model1 top10000 avg_accuray cov_accuracy
model2 top10000 avg_accuray cov_accuracy
