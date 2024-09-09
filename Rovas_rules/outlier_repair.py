"""
常用的异常值检测修复方式如下：
一、删除异常值
如果由于数据输入错误、数据处理错误或异常值观测值非常小，我们会删除异常值。
我们还可以在两端使用修剪来去除异常值。但是当数据集很小的时候不适合删除异常值
二、转换值
# 这些技术将数据集中的值转换为更小的值。
如果数据有很多极端值或倾斜，此方法有助于使您的数据正常。但是这些技巧并不总是给你最好的结果。从这些方法中不会丢失数据。
三、插补法
我们可以使用平均值、中位数、零值。由于我们进行了输入，所以没有丢失数据。
这里的中值是合适的，因为它不受异常值的影响。
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy

# section 一、删除异常值

train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')
sns.boxplot(train['Cost of Living Index'])
plt.title("Box Plot before outlier removing")
plt.show()
def drop_outliers(df, field_name):
    iqr = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
    df.drop(df[df[field_name] > (iqr + np.percentile(df[field_name], 75))].index, inplace=True)
    df.drop(df[df[field_name] < (np.percentile(df[field_name], 25) - iqr)].index, inplace=True)
drop_outliers(train, 'Cost of Living Index')
sns.boxplot(train['Cost of Living Index'])
plt.title("Box Plot after outlier removing")
plt.show()

# section 二、转换值

# Box-transformation
train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')
sns.boxplot(train['Rent Index'])
plt.title("Box Plot before outlier removing")
plt.show()
train['Rent Index'],fitted_lambda= scipy.stats.boxcox(train['Rent Index'] ,lmbda=None)
sns.boxplot(train['Rent Index'])
plt.title("Box Plot after outlier removing")
plt.show()

# section 三、插补法

# median imputation
train = pd.read_csv('../input/titanic/train.csv')
sns.boxplot(train['Age'])
plt.title("Box Plot before median imputation")
plt.show()
q1 = train['Age'].quantile(0.25)
q3 = train['Age'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(train['Age'])
for i in train['Age']:
    if i > Upper_tail or i < Lower_tail:
            train['Age'] = train['Age'].replace(i, med)
sns.boxplot(train['Age'])
plt.title("Box Plot after median imputation")
plt.show()