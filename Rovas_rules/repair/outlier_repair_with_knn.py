"""
使用knn处理无穷大的值和缺失值（将无穷大的值标记为缺失值）
可处理特征列和标签列，通过选择邻居，将邻居在该特征处的均值或中位数作为缺失值的值
KNeighborsRegressor 主要用于回归任务，它预测的是连续值。
对于标签分类任务，通常会使用 KNeighborsClassifier
"""
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepod.models.tabular import GOAD, RoSAS
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import IsolationForest

# 示例数据
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
data['Class'] = enc.fit_transform(data['Class'])

# section 识别和修复原始数据集中的缺失值和无穷大的值

# 处理无穷大值
# 将无穷大值替换为 np.nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# 使用 KNN 修复缺失值
knn_imputer = KNNImputer(n_neighbors=2)
data_imputed = knn_imputer.fit_transform(data)

# 转换回 DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
print("修复后的数据：")
print(data_imputed)

# section 识别和修复标签中的异常值

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# subsection 真实数据集且对数据集的特征进行了处理

X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
class_names = enc.classes_
feature_names = data.columns.values.tolist()
# 将 X 和 y 组合为一个 numpy 数组
combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 创建新的 DataFrame
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 对分类特征进行整数编码
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_

# subsection 针对元组异常的无监督异常检测器GOAD

# 无监督检测器GOAD
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# 弱监督检测器RoSAS
# 使用异常检测模型识别异常样本
# IsolationForest 通过分析样本的特征来检测异常值，并不会直接处理标签。它基于样本的孤立性质来判断异常
iso_forest = IsolationForest(contamination=0.1)
y_pred_anomaly = iso_forest.fit_predict(X_train)
# 标签 -1 表示异常，1 表示正常
anom_id = np.where(y_pred_anomaly == -1)[0]
clf_deep = RoSAS(epochs=1, hidden_dims=20,
                   device=device,
                   random_state=42)
# # 获取所有类别索引，假设 `1` 是一个正常类别，其他类别为异常
# anom_id = np.where(y_train == 1)[0]
# 随机选择10个异常样本的索引
known_anom_id = np.random.choice(anom_id, 10, replace=False)
# 创建标签数组，初始化为0
y_semi = np.zeros_like(y_train, dtype=int)
# 设置已知异常样本的标签为1
y_semi[known_anom_id] = 1
# 训练模型
clf_deep.fit(X_train, y_semi)

# subsection 借助异常检测器，在训练集上进行异常值检测，输出异常值索引。检测的不是异常标签，而是异常特征
# subsection 检测的是特征中的异常值索引，后续需进行outliers_index_numpy的替换，使其对应异常的标签
clf = clf_deep
train_scores = clf.decision_function(X_train)
train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
train_outliers_index = []
train_outliers_confidence = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
        train_outliers_confidence.append(train_confidence[i])
print("训练集中异常值索引：", train_outliers_index)
outliers_index_numpy = np.array(train_outliers_index)

# subsection 借助KNN进行异常标签的修复

X_inners = X_train[~outliers_index_numpy]
y_inners = y_train[~outliers_index_numpy]
# KNN模型
# knn = KNeighborsRegressor(n_neighbors=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_inners, y_inners)
# 预测异常值
X_outliers = X_train[outliers_index_numpy]
y_pred = knn.predict(X_outliers)
print("待修复的标签:", y_train[outliers_index_numpy])
print("修复后的标签:", y_pred)
# 替换异常值
y_train[outliers_index_numpy] = y_pred
# print("修复后的数据标签:", y_train)