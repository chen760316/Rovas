import pandas as pd
import numpy as np

# 读取 .tes 文件
tes_df = pd.read_csv('vowels/ae.train', delimiter='\s+')  # 根据需要选择分隔符
X_test = tes_df.iloc[:, :-1].values  # 前 n-1 列
feature_names = [f'feature_{i+1}' for i in range(X_test.shape[1])]
column_names = feature_names + ['label']
tes_df.columns = column_names

# 读取 .tra 文件
tra_df = pd.read_csv('vowels/ae.test', delimiter='\s+')  # 根据需要选择分隔符
tra_df.columns = column_names

# 合并 DataFrame
combined_df = pd.concat([tra_df, tes_df], ignore_index=True)

# 保存为 CSV 文件
combined_df.to_csv('vowels/vowels.csv', index=False)