import pandas as pd
import numpy as np

df = pd.read_csv('E:/Rovas/Rovas_rules/baselines/datasets/uadd/speech-unsupervised-ad.tab', sep='\t')

X = df.iloc[:, :-1].values  # 前 n-1 列
y = df.iloc[:, -1].values    # 最后一列
y = y.reshape(-1, 1)
combined_numpy = np.hstack((X, y))
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
column_names = feature_names + ['label']
# 转换为 DataFrame
df = pd.DataFrame(combined_numpy, columns=column_names)

# if 'id' in df.columns:
#     df = df.drop(columns=['id'])
# else:
#     print("列 'id' 不存在于数据中")

df.to_csv('E:/Rovas/Rovas_rules/baselines/multi_class_datasets/open_source_data/speech.csv', index=False)