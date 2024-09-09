"""
𝑅(𝑡) ∧ M𝑜 (𝑡, D) ∧ 𝑋1 → ugly(𝑡)的具体实现
分类器为linear核的svm分类器
"""
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

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

def calculate_made(data):
    median = np.median(data)  # 计算中位数
    abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
    mad = np.median(abs_deviation)  # 计算绝对误差均值
    made = 1.843 * mad
    return median, made

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# section 数据预处理
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
# print("X_train 原始索引:", train_indices)
# print("X_test 原始索引:", test_indices)
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

# SECTION M𝑜 (𝑡, D)
# subsection 针对元组异常的无监督异常检测器GOAD
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# subsection 针对元组异常的弱监督异常检测器DeepSAD
clf_deep = DeepSAD(epochs=1, hidden_dims=20,
                   device=device,
                   random_state=42)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf_deep.fit(X_train, y_semi)

# SECTION 借助异常检测器，在训练集上进行异常值检测
clf = clf_gold
train_scores = clf.decision_function(X_train)
train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))

# SECTION 谓词loss(M, D, 𝑡)的实现
# SUBSECTION SVM模型的实现
# svm_model = svm.SVC()
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
# 训练样本中的异常值索引
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
# 训练样本中的异常值索引
bad_samples = np.where(hinge_loss > 1)[0]
print("损失函数高于损失阈值的样本索引为：", bad_samples)

# subsection 原始数据中的svm分类准确度
print("*" * 100)
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
print("*" * 100)

# SECTION M𝑐 (𝑅, 𝐴, M)
top_k_svm = 6
# 提取系数
feature_importances_coef = np.abs(svm_model.coef_[0])
# 对系数进行排序
top_k_indices = np.argpartition(-feature_importances_coef, top_k_svm)[:top_k_svm]
print("SVM模型选择的特征索引是：", top_k_indices)
important_features_coef = top_k_indices

# section imbalanced(𝐷, 𝑅, 𝑡 .𝐴, 𝛿)，适用于整个输入数据𝐷
from sklearn.preprocessing import MinMaxScaler
# 设置分组的间隔
interval = 0.01
# 初始化MinMaxScaler
scaler = MinMaxScaler()
col_indices = 10
row_indices = 100
train_row_number = X_train.shape[0]
select_feature = feature_names[col_indices]
data_imbalance = pd.read_excel(file_path)
data_imbalance[data.columns] = scaler.fit_transform(data[data.columns])
ta = data_imbalance.iloc[row_indices, col_indices]
# 对每列数据进行分组
bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
digitized = np.digitize(data_imbalance[select_feature], bins)
# 统计每个区间的计数
unique_bins, counts = np.unique(digitized, return_counts=True)
# 找到 ta 所在的间隔
ta_bin = np.digitize([ta], bins)[0]
# 找到 ta 所在间隔的计数
ta_count = counts[unique_bins == ta_bin][0]
# 设置最小支持数差值
median_imbalance, made_imbalance = calculate_made(counts)
lower_threshold = median_imbalance - 2 * made_imbalance
upper_threshold = median_imbalance + 2 * made_imbalance
if ta_count < lower_threshold or ta_count > upper_threshold:
    print("所选列A在所选元组t处是不平衡的")
else:
    print("所选列A在所选元组t处是平衡的")

# section SDomain(𝐷, 𝑅, 𝐴, 𝜎)
from sklearn.preprocessing import MinMaxScaler
# 设置分组的间隔
interval = 0.01
col_indices = 10
selected_bins = 0
columns_bins = {}
columns_bins_count = []
# 初始化MinMaxScaler
scaler = MinMaxScaler()
select_feature = feature_names[col_indices]
data_minmax = pd.read_excel(file_path)
data_minmax[data.columns] = scaler.fit_transform(data[data.columns])
# 对每列数据进行分组
bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
# 统计每列数据占据了多少个间隔
for column in data_minmax.columns:
    digitized = np.digitize(data_minmax[column], bins)
    unique_bins, counts = np.unique(digitized, return_counts=True)
    print(f"列 '{column}' 占据了 {len(unique_bins)} 个间隔")
    columns_bins[column] = len(unique_bins)
    columns_bins_count.append(len(unique_bins))
    if column == select_feature:
        selected_bins = len(unique_bins)
median, made = calculate_made(np.array(columns_bins_count))
lower_threshold = median - 2 * made
upper_threshold = median + 2 * made
if selected_bins < lower_threshold:
    print("所选列A的活动域过小")
else:
    print("所选列A的活动域正常")

# section 确定所选属性A下元组t是否为ugly outliers
# 异常检测器检测到的异常值 (训练集中所有的outliers)
outliers_index_numpy = np.array(train_outliers_index)
# 异常值中导致SVM分类错误的部分（筛选出训练集中所有的bad outliers）
intersection = np.intersect1d(outliers_index_numpy, bad_samples)
# 映射回原始数据集对应的元组索引
initial_intersection = train_indices[intersection]
# 在这部分分类错误的异常值中继续引入谓词 (在bad outliers的基础上进一步区分出ugly outliers)
influential_features = important_features_coef
# subsection 判断哪些influential features的活动域过小 （SDomain(𝐷, 𝑅, 𝐴, 𝜎)）
feature_with_small_domain = []
for feature in influential_features:
    select_feature = feature_names[feature]
    digitized = np.digitize(data_minmax[select_feature], bins)
    unique_bins, counts = np.unique(digitized, return_counts=True)
    selected_bins = len(unique_bins)
    if selected_bins < lower_threshold:
        feature_with_small_domain.append(feature)
print("有影响力的特征中活动域过小的特征索引为：", feature_with_small_domain)
# subsection 判断活动域过小的特征中哪些元组为不平衡的元组（imbalanced(𝐷, 𝑅, 𝑡 .𝐴, 𝛿)）
selected_tuples = []
for small_feature in feature_with_small_domain:
    for row_index in original_indices:
        ta = data_imbalance.iloc[row_index, small_feature]
        select_feature = feature_names[small_feature]
        # 对每列数据进行分组
        # 生成0-1之间100个间隔的数组
        bins = np.arange(0, 1.01, interval)
        digitized = np.digitize(data_imbalance[select_feature], bins)
        # 统计每个区间的计数
        unique_bins, counts = np.unique(digitized, return_counts=True)
        # 找到 ta 所在的间隔
        ta_bin = np.digitize([ta], bins)[0]
        # 找到 ta 所在间隔的计数
        ta_count = counts[unique_bins == ta_bin][0]
        # 设置最小支持数差值
        median, made = calculate_made(counts)
        lower_threshold = median - 2 * made
        upper_threshold = median + 2 * made
        if ta_count < lower_threshold or ta_count > upper_threshold:
            selected_tuples.append(row_index)
# ugly_outliers = set(selected_tuples).union(set(initial_intersection))
ugly_outliers = set(selected_tuples).intersection(set(initial_intersection))
print("bad outliers为：", set(initial_intersection))
print("ugly outliers为：", ugly_outliers)