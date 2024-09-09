"""
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 ) ∧ loss(M, D, 𝑡) > 𝜆 ∧ M𝑐 (𝑅, 𝐴, M) → ugly(𝑡)的具体实现
分类器为不是linear核的svm分类器
"""
# unsupervised methods
from collections import Counter

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

# section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现（在训练数据D的所有元组中找异常值）
threshold = 0.01
col_indices = 3
row_indices = 10
select_feature = feature_names[col_indices]
# 获得所选列的数据
select_column_data = data_copy[select_feature].values
# # 确定训练集的行数
# num_rows_X_train = X_train.shape[0]
# # 截取 select_column_data 中X_train下标对应的元素
# select_column_data_trimmed = select_column_data[train_indices]
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
# 所有数据D下对应的下标索引
outliers_index_numpy = np.array(outliers_index)
print("A属性下所有异常值的索引为：", outliers_index)
print("A属性下所有异常值为：", outliers)

# SECTION 谓词loss(M, D, 𝑡)的实现

#  SVM模型训练和分类准确度
# svm_model = svm.SVC()
svm_model = svm.SVC(probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
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
# 在训练数据中判定的被错误分类的样本的索引
bad_samples = np.where(hinge_loss > 1)[0]
print("损失函数高于损失阈值的样本索引为：", bad_samples)

# subsection 原始数据中的svm分类准确度
print("*" * 100)
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
print("*" * 100)

# section 谓词 M𝑐 (𝑅, 𝐴, M) 的实现
# SUBSECTION 借助方差判别有影响力的特征
top_k_var = 6
variances = np.var(X_train, axis=0)
top_k_indices_var = np.argsort(variances)[-top_k_var:][::-1]
print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices_var))

# SUBSECTION sklearn库的SelectKBest选择器，借助Fisher检验筛选最有影响力的k个特征
top_k_fisher = 6
selector = SelectKBest(score_func=f_classif, k=top_k_fisher)
y_trans_fisher = y_train.reshape(-1)
X_new = selector.fit_transform(X_train, y_trans_fisher)
# 获取被选中的特征的索引
selected_feature_indices = selector.get_support(indices=True)
print("SelectKBest选择器借助Fisher检验的最有影响力的{}个特征的索引：{}".format(top_k_fisher, selected_feature_indices))

# SUBSECTION 无模型(非参数)方法中的Permutation Feature Importance-slearn(很耗时)
# top_k_svm = 6
# result = permutation_importance(svm_model, X_train, y_train, n_repeats=10,random_state=42)
# permutation_importance = result.importances_mean
# top_k_permutation = np.argpartition(-permutation_importance, top_k_svm)[:top_k_svm]
# print("Permutation Feature Importance-slearn检验的最有影响力的的前{}个属性的索引：{}".format(top_k_svm, top_k_permutation))

# SUBSECTION LIME(Local Interpretable Model-Agnostic Explanation)，需要借助XGBoost
i = 16
top_k_svm = 6
np.random.seed(1)
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
# predict_proba 方法用于分类任务，predict 方法用于回归任务
predict_fn = lambda x: svm_model.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=6)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
important_features = []
for feature_set in top_features:
    feature_long = feature_set[0]
    for feature in feature_names:
        if set(feature).issubset(set(feature_long)):
            important_features.append(feature)
            break
important_feature_indices = [feature_names.index(feature_name) for feature_name in important_features]
print("LIME检验的最有影响力的的前{}个属性的索引：{}".format(top_k_svm, important_feature_indices))

# section 确定最重要的特征

# subsection 仅采用基于方差的方法
important_features_var = top_k_indices_var

# subsection 仅采用基于Fisher检验的方法
important_features_fisher = selected_feature_indices

# subsection 仅采用Permutation Importance的方法
# important_features_perm = top_k_permutation

# subsection 仅采用基于LIME的方法
important_features_lime = important_feature_indices

# subsection 将上述所有方法返回的重要特征按出现次数由高到低得到最重要的K个特征
# 将所有 ndarrays 合并到一个列表中
# all_indices = np.concatenate([important_features_var, important_features_fisher,
#                               important_features_perm, important_features_lime])
all_indices = np.concatenate([important_features_var, important_features_fisher, important_features_lime])
# 统计每个特征出现的次数
index_counts = Counter(all_indices)
# 按出现次数从高到低排序
sorted_indices = sorted(index_counts.items(), key=lambda x: x[1], reverse=True)
# 提取排序后的特征索引
sorted_feature_indices = [index for index, count in sorted_indices]
# 选择前 k 个最重要的特征
k = 6
important_features_mix = sorted_feature_indices[:k]
print("所有特征索引及其出现次数:", index_counts)
print("按频次排序的特征索引:", sorted_indices)
print("频次计数方法选择的最重要的前 {} 个特征索引: {}".format(k, important_features_mix))

# section 确定所选属性A下元组t是否为ugly outliers
# 在原始数据D中的下标索引
intersection = np.intersect1d(outliers_index_numpy, train_indices[bad_samples])
influential_features = important_features_mix
if np.isin(row_indices, intersection) and np.isin(col_indices, influential_features):
    print("所选元组t在属性A下是ugly outliers")
elif np.isin(row_indices, intersection) and np.logical_not(col_indices, influential_features):
    print("所选元组t在属性A下不是ugly outliers，是bad outliers")
else:
    print("所选元组t在属性A下既不是ugly outliers，也不是bad outliers")

if np.isin(col_indices, influential_features):
    print("A特征是有影响力的特征")
else:
    print("A特征不是有影响力的特征")

# section 确定所有有影响力的特征下的所有ugly outliers
ugly_outlier_list = {}
for column in influential_features:
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
    # 在原始数据集D下的下标索引，同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆
    intersection = np.intersect1d(outliers_index_numpy, train_indices[bad_samples])
    ugly_outlier_list[column] = intersection

for idx, key in enumerate(ugly_outlier_list):
    value = ugly_outlier_list[key]
    print(f"第 {key} 列的ugly outliers为: {value}")