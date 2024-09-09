"""
ugly outliers的修复，分类器为linear核的svm分类器
输入数据集为deepod.utils.data中generate_data方法下的合成数据集
适用于二分类的情况
数据集过于简单，SVM模型能无错误地进行分类
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

# subsection 合成数据集
from deepod.utils.data import generate_data
n_train = 500
n_test = 100
n_features = 10
contamination = 0.1
epochs = 1
#      正常值由多元高斯分布生成，
#      异常值是由均匀分布生成的。
X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train, n_test=n_test, n_features=n_features,
    contamination=contamination, random_state=random_state)

# SECTION M𝑜 (𝑡, D)
# subsection 针对元组异常的无监督异常检测器GOAD
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# SECTION 借助异常检测器，在训练集上进行异常值检测
clf = clf_gold
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
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))
print("训练集中的异常值的置信度：", train_outliers_confidence)

# SECTION SVM模型的实现
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
# 被SVM模型错误分类的样本
# 训练样本中的异常值索引
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]

# SUBSECTION 使用sklearn库中的hinge损失函数
decision_values = svm_model.decision_function(X_train)
# 计算每个样本的hinge损失
num_samples = X_train.shape[0]
num_classes = svm_model.classes_.shape[0]
hinge_loss = np.zeros(num_samples)
for i in range(num_samples):
    hinge_loss[i] = np.maximum(0, 1 - (2 * y_train[i] - 1) * decision_values[i])
# 判定异常：设置阈值为 1，超过此值即认为是异常
# 训练样本中的异常值索引
bad_samples = np.where(hinge_loss > 1)[0]
correct_samples = np.where(hinge_loss <= 1)[0]
# print("损失函数高于损失阈值的样本索引为：", bad_samples)

# subsection 判定训练数据中异常值可能导致分类错误的样本
# 训练数据中的异常值，导致hinge损失大于1的样本
inter_outliers = list(set(train_outliers_index) & set(bad_samples))
# 训练数据中的异常值，导致hinge损失小于等于1的样本
inter_correct_class = list(set(train_outliers_index) & set(correct_samples))
# 训练数据中的hinge损失大于1且被SVM误分类的样本
intersection = np.intersect1d(bad_samples, wrong_classified_indices)
# 训练数据中被SVM误分类且hinge损失小于等于1的样本
diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("训练数据中被SVM误分类且hinge损失小于等于1的样本：", diff_elements)
print("训练集中是异常值且hinge损失高于阈值1的样本索引：", inter_outliers)
print("训练集中是异常值且hinge损失小于等于1的样本索引：", inter_correct_class)

# SECTION 原始数据中的svm在各类评价指标下的表现
print("*" * 100)

# subsection 计算SVM的分类准确度
# 准确度是指模型正确分类的样本数占总样本数的比例
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))

# subsection 计算 F1 分数
f1_train = f1_score(y_train, svm_model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, svm_model.predict(X_test), average='weighted')
print("原始训练集SVM分类F1分数：" + str(f1_train))
print("原始测试集SVM分类F1分数：" + str(f1_test))

# SECTION 舍弃掉训练数据中hinge损失函数高于1且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)
# subsection 计算 accuracy分数
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_outliers] = False
# 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练SVM模型
svm_model_split = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_split.fit(X_train_split, y_train_split)
print("去除异常值中分类错误的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, svm_model_split.predict(X_train_split))))
print("去除异常值中分类错误的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_split.predict(X_test))))

# subsection 计算 F1 分数
f1_train_split = f1_score(y_train_split, svm_model_split.predict(X_train_split), average='weighted')
f1_test_split = f1_score(y_test, svm_model_split.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_split))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_split))

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# subsection 计算 accuracy分数
mask_h = np.ones(len(X_train), dtype=bool)
mask_h[bad_samples] = False
X_train_h = X_train[mask_h]
y_train_h = y_train[mask_h]
svm_model_h = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_h.fit(X_train_h, y_train_h)
print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))

# subsection 计算 F1 分数
f1_train_h = f1_score(y_train_h, svm_model_h.predict(X_train_h), average='weighted')
f1_test_h = f1_score(y_test, svm_model_h.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_h))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_h))

# SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM
print("*" * 100)

# subsection 计算 accuracy分数
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
svm_model_o = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_o.fit(X_train_o, y_train_o)
print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))

# subsection 计算 F1 分数
f1_train_o = f1_score(y_train_o, svm_model_o.predict(X_train_o), average='weighted')
f1_test_o = f1_score(y_test, svm_model_o.predict(X_test), average='weighted')
print("去除异常值中分类错误的样本后的训练集SVM分类F1分数：" + str(f1_train_o))
print("去除异常值中分类错误的样本后的测试集SVM分类F1分数：" + str(f1_test_o))