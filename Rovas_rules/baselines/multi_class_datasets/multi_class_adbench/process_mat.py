import scipy.io
import numpy as np
import pandas as pd

# 读取 .mat 文件
mat_data = scipy.io.loadmat('Cardiotocography/cardio.mat')
X, y = mat_data['X'], mat_data['y']

# 使用 numpy.unique 找到唯一值
unique_values = np.unique(y)
# 计算唯一值的数量
num_unique_values = len(unique_values)
print("y属性不同的标签数量：", num_unique_values)

# 生成csv文件格式
y = y.reshape(-1, 1)
combined_numpy = np.hstack((X, y))
# 生成列名
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
column_names = feature_names + ['label']
# 转换为 DataFrame
df = pd.DataFrame(combined_numpy, columns=column_names)
# 保存为 CSV 文件
df.to_csv('Cardiotocography/cardiotocography.csv', index=False)