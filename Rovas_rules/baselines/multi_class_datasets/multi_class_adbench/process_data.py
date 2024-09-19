import pandas as pd
import numpy as np

df = pd.read_csv('yeast/yeast.data', sep='\s+')
df = df.drop(df.columns[0], axis=1)
X = df.iloc[:, :-1].values  # 前 n-1 列
y = df.iloc[:, -1].values    # 最后一列
y = y.reshape(-1, 1)
combined_numpy = np.hstack((X, y))
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
column_names = feature_names + ['label']
# 转换为 DataFrame
combined_df = pd.DataFrame(combined_numpy, columns=column_names)
combined_df.to_csv('yeast/yeast.csv', index=False)