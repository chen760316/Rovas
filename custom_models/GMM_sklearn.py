"""
sklearn库中的高斯混合模型(GMM)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import sys

sys.path.append('E:/xuhongzuo/Rovas/')

# 生成示例数据
np.random.seed(0)
# 生成三个二维高斯分布的数据
# X1 = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], 1000)
# X2 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
# X3 = np.random.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], 1000)
# X = np.concatenate([X1, X2, X3])

# 生成两个特征的随机数据
# X = np.random.rand(1000, 2)

# 借助make_blobs库合成数据
feature_number = 5
X, _ = make_blobs(n_samples=1000, centers=3, n_features=feature_number, random_state=None)

# 添加噪声
# new_rows = np.random.rand(50, feature_number)
# new_rows = np.random.exponential(scale=1, size=(50, feature_number))
file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
new_rows = apple_quality.iloc[:50, 1:6].values

# 将新数组与原始数组堆叠
X_change = np.vstack((X, new_rows))
# X = X.reshape(-1, 1)

# 使用GMM拟合数据
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X_change)

# 计算每个数据点属于模型分布的概率
probabilities = gmm.predict_proba(X_change)

# 设定阈值，识别不符合模型分布的数据点
threshold = 0.8  # 根据实际情况调整
outliers_indices = np.where(np.max(probabilities, axis=1) < threshold)[0]

# 打印不符合模型分布的数据点
print("Outliers indices:", outliers_indices)

# 预测每个样本的类别
labels = gmm.predict(X)

# 绘制数据和模型
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()