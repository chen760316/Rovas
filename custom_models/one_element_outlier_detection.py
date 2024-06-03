"""
单元素检测模型（用于检测单列中的异常值）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM
import sys

sys.path.append('E:/xuhongzuo/Rovas/')

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

# 训练One-class SVM模型
model = OneClassSVM(nu=0.05)  # nu是一个超参数，用于控制异常点的比例，可以根据具体情况调整
model.fit(X_change)

# 预测异常值
pred = model.predict(X_change)
outliers = np.where(pred == -1)[0]  # 找出被预测为异常值的样本

print("异常值：", outliers)