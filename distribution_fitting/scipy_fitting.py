"""
除了正态分布，Numpy还支持许多其他分布函数，如高斯混合模型、指数分布、卡方分布、正态分布等。
备注：需要在python2.7的环境下使用，python3下会报错(python3下定义包装器使得预测值和实际值尽可能接近)
"""
import numpy as np
from scipy.stats import norm, chisquare
import matplotlib.pyplot as plt

# 定义chisquare的小包装器，使得两个输入的和(几乎)相同
def cs(n, y):
    return chisquare(n, np.sum(n)/np.sum(y) * y)

# 产生含噪声的数据集
x = np.linspace(-5, 5, num=100)
y = 3 * norm.pdf(x, loc=-1.5) + 5 * norm.pdf(x, loc=1) + np.random.normal(size=len(x))

# 拟合数据
(mu, sigma) = norm.fit(y)
# 计算Chi-squared检验的p值
expected = norm.pdf(x, mu, sigma)
observed = y
(chi2, p) = cs(observed, expected)
print("Chi-squared statistic:", chi2)
print("p-value:", p)

# 绘制拟合曲线及原始数据
plt.plot(x, y, 'o', label='Data')
plt.plot(x, norm.pdf(x, mu, sigma), label='Fitted curve')
plt.legend()
plt.show()

"""
利用scipy进行线性插值
"""
from scipy import interpolate

# 已知数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])
# 定义线性插值函数
f = interpolate.interp1d(x, y, kind='linear')
# 插值点
xnew = np.linspace(0, 5, num=100, endpoint=True)
# 使用插值函数计算插值点的值
ynew = f(xnew)
# 绘制原始数据点和插值曲线
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

"""
利用scipy进行曲线拟合
"""
from scipy.optimize import curve_fit

# 定义要拟合的函数形式，这里我们使用一个简单的二次函数作为示例
def func(x, a, b, c):
    return a * x**2 + b * x + c

# 已知数据点
xdata = np.linspace(-10, 10, 100)
y = func(xdata, 1.3, 0.1, 0.1)
ynoise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + ynoise

# 使用curve_fit进行曲线拟合
popt, pcov = curve_fit(func, xdata, ydata)

# 绘制原始数据点和拟合曲线
plt.plot(xdata, ydata, 'b.', label='data')
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.show()
