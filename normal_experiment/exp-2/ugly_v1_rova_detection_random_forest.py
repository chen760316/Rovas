"""
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧ loss(M, D, 𝑡) > 𝜆 ∧ M𝑐 (𝑅, 𝐴,M) → ugly(𝑡)
Rovas在传统异常检测领域检测数据中的异常值的效果
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# choice 选取数据集
# file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"

data = pd.read_csv(file_path)

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 20000:
    data = data.sample(n=20000, random_state=42)

enc = LabelEncoder()
label_name = data.columns[-1]

# 原始数据集D对应的Dataframe
data[label_name] = enc.fit_transform(data[label_name])

# 检测非数值列
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# 为每个非数值列创建一个 LabelEncoder 实例
encoders = {}
for column in non_numeric_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

X = data.values[:, :-1]
y = data.values[:, -1]

all_columns = data.columns.values.tolist()
feature_names = all_columns[:-1]
class_name = all_columns[-1]

# 统计不同值及其数量
unique_values, counts = np.unique(y, return_counts=True)
# 输出结果
for value, count in zip(unique_values, counts):
    print(f"标签: {value}, 数量: {count}")
# 找到最小标签的数量
min_count = counts.min()
total_count = counts.sum()

# 计算比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")
min_count_index = np.argmin(counts)  # 找到最小数量的索引
min_label = unique_values[min_count_index]  # 对应的标签值

# 找到分类特征的列名
categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
# 获取分类特征对应的索引
categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

# section 数据特征缩放以及添加噪声

# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
# 加入随机噪声的比例
noise_level = 0.2
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从加噪数据中生成加噪训练数据和加噪测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

# section 找到有影响力的特征 M𝑐 (𝑅, 𝐴, M)
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
import re

i = len(feature_names)
np.random.seed(1)
categorical_names = {}
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
rf_model.fit(X_train_copy, y_train)

for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data_copy.iloc[:, feature])
    data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
    categorical_names[feature] = le.classes_

explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

predict_fn = lambda x: rf_model.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names)//2)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
top_k_indices = [feature_names.index(name) for name in top_feature_names]
print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# section 找到loss(M, D, 𝑡) > 𝜆的元组

# # choice 使用sklearn库中的hinge损失函数
# decision_values = rf_model.decision_function(X_copy)
# predicted_labels = np.argmax(decision_values, axis=1)
# # 计算每个样本的hinge损失
# num_samples = X_copy.shape[0]
# num_classes = rf_model.classes_.shape[0]
# hinge_losses = np.zeros((num_samples, num_classes))
# hinge_loss = np.zeros(num_samples)
# for i in range(num_samples):
#     correct_class = int(y[i])
#     for j in range(num_classes):
#         if j != correct_class:
#             loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
#             hinge_losses[i, j] = loss_j
#     hinge_loss[i] = np.max(hinge_losses[i])
#
# # 在所有加噪数据D中损失函数高于阈值的样本索引
# ugly_outlier_candidates = np.where(hinge_loss > 1)[0]
# # print("D中损失函数高于损失阈值的样本索引为：", ugly_outlier_candidates)

# choice 使用交叉熵损失函数
from sklearn.preprocessing import OneHotEncoder

# 获取概率值
y_pred = rf_model.predict_proba(X_copy)[:, [1, 0]]
# 创建 OneHotEncoder 实例
encoder = OneHotEncoder(sparse=False)
# 拟合并转换 y_test
y_true = encoder.fit_transform(y.reshape(-1, 1))
# 计算每个样本的损失
loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# 计算测试集平均多分类交叉熵损失
average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
bad_samples = np.where(loss_per_sample < average_loss)[0]
good_samples = np.where(loss_per_sample >= average_loss)[0]
ugly_outlier_candidates = bad_samples

# section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现，找到所有有影响力的特征下的异常元组

outlier_feature_indices = {}
threshold = 0.01
for column_indice in top_k_indices:
    select_feature = feature_names[column_indice]
    select_column_data = data_copy[select_feature].values
    max_value = np.max(select_column_data)
    min_value = np.min(select_column_data)
    sorted_indices = np.argsort(select_column_data)
    sorted_data = select_column_data[sorted_indices]
    # 找到A属性下的所有异常值
    outliers = []
    outliers_index = []
    # 检查列表首尾元素
    if len(sorted_data) > 1:
        if (sorted_data[1] - sorted_data[0] >= threshold):
            outliers.append(sorted_data[0])
            outliers_index.append(sorted_indices[0])
        if (sorted_data[-1] - sorted_data[-2] >= threshold):
            outliers.append(sorted_data[-1])
            outliers_index.append(sorted_indices[-1])
    # 检查中间元素
    for i in range(1, len(sorted_data) - 1):
        current_value = sorted_data[i]
        left_value = sorted_data[i - 1]
        right_value = sorted_data[i + 1]
        if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
            outliers.append(current_value)
            outliers_index.append(sorted_indices[i])
    outliers_index_numpy = np.array(outliers_index)
    intersection = np.intersect1d(np.array(outliers_index), ugly_outlier_candidates)
    # print("有影响力的特征A下同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆的所有异常值索引为：", intersection)
    outlier_feature_indices[column_indice] = intersection
# print(outlier_feature_indices)

# section 确定数据中需要修复的元组

outlier_tuple_set = set()
for value in outlier_feature_indices.values():
    outlier_tuple_set.update(value)
X_copy_repair_indices = list(outlier_tuple_set)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# section 汇总加噪样本中检测到的outliers

y_pred = np.zeros_like(y)
y_pred[X_copy_repair_indices] = 1
y_train_pred = y_pred[train_indices]
y_test_pred = y_pred[test_indices]
# 统计不同值及其数量
unique_values, counts = np.unique(y_test_pred, return_counts=True)
# 输出结果
for value, count in zip(unique_values, counts):
    print(f"预测的测试集标签: {value}, 预测的标签数量: {count}")
# 找到预测的最小标签的数量
min_count = counts.min()
total_count = counts.sum()
# 计算预测的最少标签的比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")

# section 使用各种评价指标评价Rovas检测到的outliers

"""Accuracy指标"""
print("*" * 100)
print("Rovas在加噪测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

"""Precision/Recall/F1指标"""
print("*" * 100)

# average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
# average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
# average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
# average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

print("Rovas在加噪测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
print("Rovas在加噪测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
print("Rovas在加噪测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

"""ROC-AUC指标"""
print("*" * 100)
roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')  # 一对多方式
print("Rovas在加噪测试集中的ROC-AUC分数：" + str(roc_auc_test))

# """PR AUC指标"""
# print("*" * 100)
# # 计算预测概率
# y_scores = 1 / (1 + np.exp(-test_scores))
# # 计算 Precision 和 Recall
# precision, recall, _ = precision_recall_curve(y_test, y_scores)
# # 计算 PR AUC
# pr_auc = auc(recall, precision)
# print("半监督异常检测器在原始测试集中的PR AUC 分数:", pr_auc)
#
# """AP指标"""
# print("*" * 100)
# # 计算预测概率
# y_scores = 1 / (1 + np.exp(-test_scores))
# # 计算 Average Precision
# ap_score = average_precision_score(y_test, y_scores)
# print("无监督异常检测器在原始测试集中的AP分数:", ap_score)