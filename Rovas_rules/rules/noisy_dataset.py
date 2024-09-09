"""
引入的噪声可以有不同的形式，常见的包括：
    注入随机噪声: 随机生成的噪声值。
    注入标签噪声: 标签（目标变量）的错误标记。
    注入特征噪声: 对特征值添加不同分布下的噪声（如高斯噪声）。
"""
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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

# section 在特征中注入高斯随机噪声
# file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# # 加入随机噪声
# noise_level = 0.1
# # 计算噪声数量
# n_samples = X.shape[0]
# n_noise = int(noise_level * n_samples)
# # 随机选择要添加噪声的样本
# noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# # 添加高斯噪声到特征
# X[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# # print("Data with noise:\n", X.head())
# # 记录原始索引
# original_indices = np.arange(len(X))
# X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
# feature_names = data.columns.values.tolist()
# combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# # 创建新的 DataFrame
# data_copy = pd.DataFrame(combined_array, columns=feature_names)
# # 训练集中添加了高斯噪声的样本在原始数据集D中的索引
# train_noise = np.intersect1d(train_indices, noise_indices)
# # 测试集中添加了高斯噪声的样本在原始数据集D中的索引
# test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# section 在标签中注入高斯随机噪声
# file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# # 定义噪声比例
# noise_level = 0.1
# # 计算噪声数量
# n_samples = y.shape[0]
# n_noise = int(noise_level * n_samples)
# # 随机选择要添加噪声的样本
# noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# # 随机生成新的标签，注意要避免生成超出标签范围的值
# y_noisy = y.copy()
# y_noisy[noise_indices] = np.random.choice(np.delete(np.unique(y), -1), n_noise)
# # y与y_noisy不同元素的位置和值
# dif_ele_indices = np.where(y_noisy != y)[0]
# dif_elements = y[dif_ele_indices]
# print("加入噪声后的标签与原标签不同的元素索引：", dif_ele_indices)
# print("加入噪声后的标签与原标签不同的元素占所有样本的比例：", len(dif_ele_indices)/len(y))
# # print("Labels with noise:\n", y_noisy.head())
# # 记录原始索引
# original_indices = np.arange(len(y))
# X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y_noisy, original_indices, test_size=0.5, random_state=1)
# feature_names = data.columns.values.tolist()
# combined_array = np.hstack((X, y_noisy.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# # 创建新的 DataFrame
# data_copy = pd.DataFrame(combined_array, columns=feature_names)
# # 训练集中添加了高斯噪声的样本在原始数据集D中的索引
# train_noise = np.intersect1d(train_indices, noise_indices)
# # 测试集中添加了高斯噪声的样本在原始数据集D中的索引
# test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# section 在特征中注入不同分布的噪声数据
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 加入随机噪声
noise_level = 0.1
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# 对特征添加噪声，例如使用均匀噪声
X[noise_indices] += np.random.uniform(-1, 1, (n_noise, X.shape[1]))
# print("Data with noise:\n", X.head())
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.5, random_state=1)
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 创建新的 DataFrame
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)
print("训练集中的噪声样本为：", train_noise)
print("测试集中的噪声样本为：", test_noise)
