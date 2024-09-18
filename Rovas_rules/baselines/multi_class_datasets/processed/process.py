import pandas as pd
import numpy as np

file_path = "../original/wine.csv"
data = pd.read_csv(file_path)
# data = pd.read_csv(file_path, sep=';')
column_names = data.columns

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 统计各个元素的数量
unique_elements, counts = np.unique(y, return_counts=True)

# 创建一个字典，将元素与其数量关联起来
element_count = dict(zip(unique_elements, counts))

# 找到数量最少的元素
min_count_element = min(element_count, key=element_count.get)

# 初始化新的标签数组
y_new = np.zeros_like(y)

# 将数量最少的元素对应的标签改为1
y_new[y == min_count_element] = 1

# 合并 X 和 y_new 为一个 NumPy 数组
combined_array = np.column_stack((X, y_new))

# 将合并后的 NumPy 数组转换为 DataFrame，并添加列名
df_combined = pd.DataFrame(combined_array, columns=column_names)

# 保存 DataFrame 为 CSV 文件
df_combined.to_csv('wine_outlier.csv', index=False)

print("DataFrame已保存为相应文件")