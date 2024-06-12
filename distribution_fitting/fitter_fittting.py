"""
Fitter参数：
data (list) –输入的样本数据；
xmin (float) – 如果为None，则使用数据最小值，否则将忽略小于xmin的数据；
xmax (float) – 如果为None，则使用数据最大值，否则将忽略大于xmin的数据；
bins (int) – 累积直方图的组数，默认=100；
distributions (list) – 给出要查看的分布列表。 如果没有，则尝试所有的scipy分布(80种),常用的分布distributions=[‘norm’,‘t’,‘laplace’,‘cauchy’, ‘chi2’,’ expon’, ‘exponpow’, ‘gamma’,’ lognorm’, ‘uniform’]；
verbose (bool) –
timeout – 给定拟合分布的最长时间，（默认=10s） 如果达到超时，则跳过该分布。
"""
from scipy import stats
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt

# N(0,2)+N(0,10)
data1 = list(stats.norm.rvs(loc=0, scale=2, size=70000))
data2 = list(stats.norm.rvs(loc=0, scale=20, size=30000))
data=np.array(data1+data2)
f = Fitter(data, distributions=['norm', 't', 'laplace'])
f.fit()
# 返回排序好的分布拟合质量（拟合效果从好到坏）,并绘制数据分布和Nbest分布
f.summary()
# 返回这些分布的拟合质量（均方根误差的和）
print(f.df_errors)
# 返回拟合分布的参数
print(f.fitted_param)
# 使用最适合数据分布的分布参数生成的概率密度
# print(f.fitted_pdf)
# 返回最佳拟合分布及其参数
print(f.get_best(method='sumsquare_error'))
# 绘制组数=bins的标准化直方图
f.hist()
plt.show()
# 绘制分布的概率密度函数
f.plot_pdf(names=None, Nbest=3, lw=2)
plt.show()


