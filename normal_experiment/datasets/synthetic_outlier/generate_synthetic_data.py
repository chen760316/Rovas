from data_generator import DataGenerator
import numpy as np
import pandas as pd

file_path = "../real_outlier/annthyroid.csv"
data = pd.read_csv(file_path)
outlier_ratio = 0.3


# data = pd.read_csv(file_path, sep=';')
column_names = data.columns

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

count_zeros = np.sum(y == 0)
print("count_zeros:", count_zeros)
num_outliers = int((outlier_ratio * count_zeros)/(1-outlier_ratio))

X_normal = X[y == 0]
y_normal = y[y == 0]

X_anomaly = X_normal.sample(n=num_outliers, random_state=42)
y_anomaly = np.ones(len(X_anomaly))

X_combined = pd.concat([X_normal, X_anomaly],ignore_index=True)
y_combined = pd.concat([y_normal, pd.Series(y_anomaly)],ignore_index=True)


# X_combined = X_combined.reset_index(drop=True)
# y_combined = y_combined.reset_index(drop=True)
# x_new = X_combined[y_combined == 0]
# print(x_new)
# # print(X_combined.shape)
# # print(y_combined.shape)
# selected_rows = y_combined[y_combined == 0].index
# print(X_combined.loc[selected_rows])
# print(X_combined.dtypes)
# print("=========")
# print(y_combined.dtypes)

# # Generate synthetic data
data_generator = DataGenerator()

# # X, y, realistic_synthetic_mode, alpha:int, percentage:float)
# # Check for NaNs


X_synthetic, y_synthetic = data_generator.generate_realistic_synthetic(X_combined, y_combined, 'local', alpha=1.5, percentage=0.2)
combined_array = np.column_stack((X_synthetic, y_synthetic))

# 将合并后的 NumPy 数组转换为 DataFrame，并添加列名
df_combined = pd.DataFrame(combined_array, columns=column_names)

# 保存 DataFrame 为 CSV 文件
df_combined.to_csv("annthyroid_0.3.csv", index=False)
count_ones = np.sum(y_synthetic == 1)

# 计算总元素的数量
total_count = len(y_synthetic)

# 计算比例
proportion_ones = count_ones / total_count
print("总元素数量为：", total_count)
print("异常值比例为：", proportion_ones)
# Save the synthetic data
# data_generator.save_data(r'D:\CodeWork\python\outlier\Rovas\Rovas_rules\baselines\multi_class_datasets\synthetic\optdigits_local_1.5_0.2.csv')