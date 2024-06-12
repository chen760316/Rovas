"""
拟合多项式分布
"""
import numpy as np
import matplotlib.pyplot as plt

"""np.polyfit函数(采用的是最小二次拟合)"""
xxx = np.arange(0, 1000)
yyy = np.sin(xxx*np.pi/180)
# 用7次多项式拟合，可改变多项式阶数
z1 = np.polyfit(xxx, yyy, 7)
# 得到多项式系数，按照阶数从高到低排列
p1 = np.poly1d(z1)
print(p1)
# 求对应xxx的各项拟合函数值
yvals=p1(xxx)

plt.plot(xxx, yyy, '*',label='original values')
plt.plot(xxx, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend在图中的位置，类似象限的位置
plt.title('polyfitting')
plt.show()

"""np.polyld函数(暂时不需要)"""
