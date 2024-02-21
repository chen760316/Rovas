import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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


