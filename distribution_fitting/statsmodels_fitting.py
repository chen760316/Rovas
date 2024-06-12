"""
model1.summary()中的参数：
R-squared：决定系数，其值=SSR/SST，SSR是Sum of Squares for Regression，SST是Sum of Squares for Total，这个值范围在[0, 1]，其值越接近1，说明回归效果越好
F-statistic：这就是我们经常用到的F检验，这个值越大越能推翻原假设，本例中其值为156.9，这个值过大，说明我们的模型是线性模型，原假设是“我们的模型不是线性模型”
Prob (F-statistic)：这就是上面F-statistic的概率，这个值越小越能拒绝原假设，本例中为1.25e-08，该值非常小了，足以证明我们的模型是线性显著的
P>|t|：统计检验中的P值，这个值越小越能拒绝原假设
t：就是我们常用的t统计量，这个值越大越能拒绝原假设
coef：指自变量和常数项的系数，本例中自变量系数是4.9193，常数项是10.2779
std err：系数估计的标准误差
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

distance = [0.7, 1.1, 1.8, 2.1, 2.3, 2.6, 3, 3.1, 3.4, 3.8, 4.3, 4.6, 4.8, 5.5, 6.1]
loss = [14.1, 17.3, 17.8, 24, 23.1, 19.6, 22.3, 27.5, 26.2, 26.1, 31.3, 31.3, 36.4, 36, 43.2]
data = pd.DataFrame({'distance': distance, 'loss': loss})

y1 = loss
X1 = distance
# 增加一个常数1，对应回归线在y轴上的截距
X1 = sm.add_constant(X1)
# 用最小二乘法建模
regression1 = sm.OLS(y1, X1)
# 数据拟合
model1 = regression1.fit()
# statsmodels.formula.api的用法
# 括号里输入公式和数据
regression2 = smf.ols(formula='loss ~ distance', data=data)
model2 = regression2.fit()
print(model1.summary())
print(model1.rsquared)
print(model1.fvalue)
print(model1.f_pvalue)