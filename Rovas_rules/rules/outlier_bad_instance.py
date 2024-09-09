"""
 𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 ) ∧ loss(M, D, 𝑡) > 𝜆 → bad(𝑡)的具体实现
"""
# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, NeuTraL, DeepSAD
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import shap
from distfit import distfit
from fitter import Fitter
import scipy.stats as stats
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 预处理数据集
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
class_names = enc.classes_
feature_names = data.columns.values.tolist()
# 使用处理后的X和y组合成新的data_copy
column_names = data.columns.tolist()
# 将 X 和 y 组合为一个 numpy 数组
combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 创建新的 DataFrame
data_copy = pd.DataFrame(combined_array, columns=column_names)
# 对分类特征进行整数编码
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_

# section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现
threshold = 0.01
col_indices = 3
row_indices = 10
select_feature = feature_names[col_indices]
# 获得所选列的数据
select_column_data = data_copy[select_feature].values
# 找到所选列的最大值和最小值
max_value = np.max(select_column_data)
min_value = np.min(select_column_data)
# 找到t.A对应的值
t_value = data_copy.iloc[row_indices, col_indices]
# 对数据进行排序
# sorted_data = np.sort(select_column_data)
sorted_indices = np.argsort(select_column_data)
sorted_data = select_column_data[sorted_indices]
# 找到最接近的比 t_value 大的值和比 t_value 小的值
greater_than_t_value = sorted_data[sorted_data > t_value]
less_than_t_value = sorted_data[sorted_data < t_value]
# 找到与t_value最接近的更大的值和更小的值
if greater_than_t_value.size > 0:
    closest_greater = greater_than_t_value[0]  # 最近的大于 t_value 的值
else:
    closest_greater = t_value
if less_than_t_value.size > 0:
    closest_less = less_than_t_value[-1]  # 最近的小于 t_value 的值
else:
    closest_less = t_value
# 判断t.A是否是异常值
if max_value == t_value:
    print("元组t在属性A上的投影是异常值吗:", t_value - closest_less > threshold)
elif min_value == t_value:
    print("元组t在属性A上的投影是异常值吗:", closest_greater - t_value > threshold)
else:
    print("元组t在属性A上的投影是异常值吗:", t_value - closest_less > threshold and t_value - closest_less > threshold)
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
# 在所有数据D上的异常值索引
outliers_index_numpy = np.array(outliers_index)
print("A属性下所有异常值的索引为：", outliers_index)
print("A属性下所有异常值为：", outliers)

# SECTION 谓词loss(M, D, 𝑡)的实现

#  SVM模型训练和分类准确度
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
# 在训练数据上错误分类的样本下标
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]
# SUBSECTION 使用sklearn库中的hinge损失函数
decision_values = svm_model.decision_function(X_train)
predicted_labels = np.argmax(decision_values, axis=1)
# 计算每个样本的hinge损失
num_samples = X_train.shape[0]
num_classes = svm_model.classes_.shape[0]
hinge_losses = np.zeros((num_samples, num_classes))
hinge_loss = np.zeros(num_samples)
for i in range(num_samples):
    correct_class = int(y_train[i])
    for j in range(num_classes):
        if j != correct_class:
            loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
            hinge_losses[i, j] = loss_j
    hinge_loss[i] = np.max(hinge_losses[i])
# 判定异常：设置阈值为 1，超过此值即认为是异常
# 训练数据上的bad outliers的索引下标
bad_samples = np.where(hinge_loss > 1)[0]
print("损失函数高于损失阈值的样本索引为：", bad_samples)

# section 确定所选属性A下元组t是否为bad outliers，确定属性A下的所有bad outliers
# 计算谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )为true和loss(M, D, 𝑡) ≤ 𝜆的元组的交集
intersection = np.intersect1d(outliers_index_numpy, train_indices[bad_samples])

if np.isin(row_indices, intersection):
    print("所选元组t在属性A下是bad outliers")
else:
    print("所选元组t在属性A下不是bad outliers")

print("A属性下所有的bad outliers的行索引为：", intersection)

# section 确定所有属性下的所有bad outliers
column_num = len(feature_names) - 1
bad_outlier_list = {}
for column in range(column_num):
    select_feature = feature_names[column]
    # 获得所选列的数据
    select_column_data = data_copy[select_feature].values
    sorted_indices = np.argsort(select_column_data)
    sorted_data = select_column_data[sorted_indices]
    # 找到所选属性下的所有异常值
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
    intersection = np.intersect1d(outliers_index_numpy, bad_samples)
    bad_outlier_list[column] = intersection
# print("所有特征列的bad outliers列表为：", bad_outlier_list)
# 打印所有特征列的bad outliers的值
for idx, key in enumerate(bad_outlier_list):
    value = bad_outlier_list[key]
    print(f"第 {key} 列的bad outliers为: {value}")