"""
检测单列异常的孤立森林模型
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import sys

sys.path.append('E:/xuhongzuo/Rovas/')

"""
共检测出53个异常值，其中47个异常值为真实异常值
"""

# 借助make_blobs库合成数据
feature_number = 5
X, _ = make_blobs(n_samples=1000, centers=3, n_features=feature_number, random_state=None)

# 添加噪声
file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
new_rows = apple_quality.iloc[:50, 1:6].values

# 将新数组与原始数组堆叠
# X_change = np.vstack((X, new_rows))[:, 0]
# X_change = X_change.reshape(-1, 1)

X_change = np.vstack((X, new_rows))

# X_change = np.vstack((X, new_rows))[:, 0:2]

# 创建孤立森林模型
model = IsolationForest(contamination=0.05)  # 可以根据实际情况调整参数
model.fit(X_change)

# 进行异常值预测
pred = model.predict(X_change)
outliers = np.where(pred == -1)[0]

print(outliers)