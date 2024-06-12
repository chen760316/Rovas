"""
拟合高斯分布
"""
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt


def func_gaosi(x, miu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - miu) ** 2 / 2 / sigma ** 2)


x = np.linspace(0, 1, 100)
y = func_gaosi(x, 0.5, 0.2)
y += np.random.normal(0., 0.02, x.shape)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 使用astropy进行高斯拟合
g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x, y)
print(g.mean.value, g.stddev.value)

