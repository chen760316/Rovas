"""
统计方法识别outliers
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np

name = ['John', 'Victor', 'Carlos', 'Leo', 'Kevin', 'Silva', 'Johnson', 'Lewis', 'George', 'Daniel', 'Harry', 'Jordan', 'James']
salary = [4000, 1000, 2000, 100000, 3500, 6000, 1500, 3000, 2500, 3600, 2100, 1700, 1600]
df = pd.DataFrame({'Name': name, 'Salary': salary})
plt.boxplot(df['Salary'])
plt.show()

"""
四分位极差法
"""
q1 = df['Salary'].quantile(0.25)
q3 = df['Salary'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_quartile = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
# print(outliers_quartile)

"""
标准偏差法
"""
mean = df.Salary.mean()
std = df.Salary.std()

upper_bound = mean + 3 * std
lower_bound = mean - 3 * std

outliers_standard = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
# print(outliers_standard)

"""
Z-分数法
"""
df['Salary_zscore'] = stats.zscore(df['Salary'])
outliers_Z_score = df[(df['Salary_zscore'] > 3) | (df['Salary_zscore'] < -3)]
print(outliers_Z_score)

"""
四分位极差法
MAD(绝对误差均值)
"""
def calculate_iqr(data):
    sorted_data = np.sort(data)  # 将数据集按升序排列
    q1 = np.percentile(sorted_data, 25)  # 计算下四分位数
    q3 = np.percentile(sorted_data, 75)  # 计算上四分位数
    iqr = q3 - q1  # 计算IQR
    return iqr

def calculate_mad(data):
    median = np.median(data)  # 计算中位数
    abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
    mad = np.mean(abs_deviation)  # 计算绝对误差均值
    return mad

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
iqr_result = calculate_iqr(data)
mad_result = calculate_mad(data)

print("IQR:", iqr_result)  # 输出结果应为4
print("MAD:", mad_result)  # 输出结果应为2.0

"""
使用MAD剔除异常值
"""
def mad_based_outlier(points, thresh=3.5): #这里设定的阈值3.5
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

points = np.array(data)
print(mad_based_outlier(points))