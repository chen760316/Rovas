"""
csaps是一个轻量级的Python包，其设计目的是实现数据的立方平滑样条（Cubic Smoothing Splines）拟合
csaps的核心是一个高效的Cubic Smoothing Splines算法，它可以处理一维到多维的数据集。算法的实现优化了计算性能，使得在大规模数据上也能快速运行。此外，通过设置smooth参数，用户可以灵活控制曲线的光滑度，以达到理想的拟合效果
"""
import numpy as np
from csaps import csaps
import matplotlib.pyplot as plt

x = np.linspace(0., 2*np.pi, 25)
y = np.sin(x) + np.random.randn(25) * 0.1
xi = np.linspace(x[0], x[-1], 151)
yi = csaps(x, y, xi, smooth=0.8)
plt.plot(x, y, "o")
plt.plot(xi, yi, "*")
plt.show()