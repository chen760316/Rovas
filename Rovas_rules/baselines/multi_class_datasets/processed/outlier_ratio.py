import pandas as pd
import numpy as np

file_path = "wine_outlier.csv"
data = pd.read_csv(file_path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 计算标签为 1 的元素的数量
count_ones = np.sum(y == 1)

# 计算总元素的数量
total_count = len(y)

# 计算比例
proportion_ones = count_ones / total_count

print("异常值比例为：", proportion_ones)