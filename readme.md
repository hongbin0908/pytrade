## pytrade是什么?
pytrade主要完成以下工作:
1. 获取股票数据
2. 使用talib库获取七天均值等特征
3. 使用sklearn/keras等机器学习库预估价格变动
4. 生成最终结果并验证

## pytrade依赖什么?
pytrade基于python, 依赖以下开源库:
1. talib
2. numpy
3. pandas
4. keras
5. sklearn
建议python3以上版本.

## pytrade代码逻辑简介
执行路径:
pytrade/run/run.py
主要模块及执行逻辑:
## main/work/conf.py:
配置逻辑, 用于对项目过程中的classifer等重要属性进行配置.
## main/work/build.py:
调用talib库信息生成ta特征.
## main/score/build.py:
调用main/score/score.py生成标注结果, 该标注结果主要有三种类型:
1. 以指定时间间隔的收盘价变动结果是否大于一进行二元标注
2. 以指定时间间隔的收盘价变动结果进行连续值标注
3. 已制定时间间隔的前一天开盘价变动结果进行连续值标注
## main/bitlize/bitlize.py:
使用决策树方式将连续值类型的ta特征变为离散特征, 作为训练样本
## main/work/selector.py:
对特征进行筛选, 提供两种筛选方式:
1. 使用正确率进行筛选
2. 使用信息上进行筛选
## main/work/model.py:
提供不同类型classifier用于模型训练
## main/model/ana.py:
提供不同效果评估方式进行模型效果预估, 包括在真实场景下的roi预估.