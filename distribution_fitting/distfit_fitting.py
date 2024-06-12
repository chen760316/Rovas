"""
.fit_transform(): 在经验数据 X 上拟合分布
.summary:获得拟合数据并测试拟合优度的所有分布的分数。
.predict():预测响应变量的概率
.model：输出分布的最佳参数
.plot(): 绘制最佳的拟合分布
"""
from distfit import distfit
import matplotlib.pyplot as plt

import numpy as np
X = np.random.normal(0, 2, [100,10])
y = [-8,-6,0,1,2,3,4,5,6]
dist = distfit(todf=True)
dist.fit_transform(X)
print(dist.summary)
dist.plot()
plt.show()