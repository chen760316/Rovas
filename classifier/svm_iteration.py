"""
使用SVM训练集求解hinge损失函数，因为测试集标签不可见
SVM和异常检测器使用相同的训练集和测试集
异常检测器直接输出在训练集上的异常值
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

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', None)        # 不限制显示宽度
np.set_printoptions(threshold=np.inf)

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
# 指定要检测的标签列类别为target_class时，样本中出现的异常值
target_class = 0

# SECTION kaggle datasets上的数据预处理
# SUBSECTION dry_bean数据集
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:,0:16]
y = data.values[:,16]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
class_names = enc.classes_
feature_names = data.columns.values.tolist()
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_

# SECTION M𝑜 (𝑡, D),针对元组异常的异常检测器
# SUBSECTION  GOAD异常检测器
clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf.fit(X_train, y=None)

# SECTION 借助异常检测器，在训练集上进行异常值检测
train_scores = clf.decision_function(X_train)
train_pred_labels, train_confidence = clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))

# SECTION SVM模型训练和分类准确度
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
wrong_classified_indices = np.where(y_train != svm_model.predict(X_train))[0]
X_train_outliers = X_train[train_outliers_index]
y_train_outliers_numpy = y_train[train_outliers_index]
y_train_outliers = pd.Series(y_train_outliers_numpy)
print("*" * 100)
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
if not y_train_outliers.empty:
    print("训练集中异常值的SVM分类准确度：" + str(accuracy_score(y_train_outliers, svm_model.predict(X_train_outliers))))
print("*" * 100)

# SECTION 使用sklearn库中的hinge损失函数
decision_values = svm_model.decision_function(X_train)
predicted_labels = np.argmax(decision_values, axis=1)
print("基于hinge损失的训练集上标签的SVM分类准确度：" + str(accuracy_score(y_train, predicted_labels)))
# 计算训练样本的平均损失
train_losses = hinge_loss(y_train, decision_values, labels=np.unique(y_train))
print("整个训练集下的平均hinge损失：", train_losses)

# section 计算训练样本的迭代损失
# 训练过程中记录每轮迭代的损失值
losses_per_iteration = []
# 开始训练迭代
interval = 30
# outlier_out_threshold为被异常检测器判定为异常值，且其hinge损失高于阈值的样本
outlier_out_threshold = [1794, 1669, 775, 3472, 1044, 535, 5400, 4637, 1821, 42, 6570, 4784, 2745, 4155, 4162, 5575, 6216, 5196, 6745, 3036, 6238, 1118, 5854, 999, 4968, 1773, 4847, 370]
for iteration in range(10):  # 假设迭代10轮
    svm_iteration = svm.SVC(max_iter=interval*(iteration+1))
    # 在每轮迭代中训练 SVM 分类器
    svm_iteration.fit(X_train, y_train)
    # 获取决策函数值
    decision_values = svm_iteration.decision_function(X_train)
    num_classes_iteration = svm_iteration.classes_.shape[0]
    # 计算每个样本的 hinge 损失
    mean_hinge_losses = hinge_loss(y_train, decision_values)
    # 将每轮迭代的损失值记录下来
    losses_per_iteration.append(mean_hinge_losses)
    num_samples = X_train.shape[0]
    num_classes = svm_iteration.classes_.shape[0]
    hinge_losses = np.zeros((num_samples, num_classes))
    hinge_loss_per_sample = np.zeros(num_samples)
    print("-"*100)
    print(f"Iteration {(iteration + 1) * interval}")
    for i in outlier_out_threshold:
        correct_class = int(y_train[i])
        for j in range(num_classes_iteration):
            if j != correct_class:
                loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
                hinge_losses[i, j] = loss_j
        hinge_loss_per_sample[i] = np.max(hinge_losses[i])
        print(f"Sample {i} has hinge losses = {hinge_loss_per_sample[i]}")
    print("-" * 100)
# 输出每轮迭代的损失值
print("*" * 100)
for i, losses in enumerate(losses_per_iteration):
    print(f"Iteration {(i + 1)*interval}: Mean hinge loss = {np.mean(losses)}")
print("*" * 100)

# section 计算每个样本的hinge损失
num_samples = X_train.shape[0]
num_classes = svm_model.classes_.shape[0]
hinge_losses = np.zeros((num_samples, num_classes))
hinge_loss_per_sample = np.zeros(num_samples)
for i in range(num_samples):
    correct_class = int(y_train[i])
    for j in range(num_classes):
        if j != correct_class:
            loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
            hinge_losses[i, j] = loss_j
    hinge_loss_per_sample[i] = np.max(hinge_losses[i])
# 判定异常：假设阈值为 1，超过此值即认为是异常
anomalies = np.where(hinge_loss_per_sample > 1)[0]
soft_anomalies = np.where((hinge_loss_per_sample > 0) & (hinge_loss_per_sample <= 1))[0]
correct_class = np.where(hinge_loss_per_sample == 0)[0]
# 输出训练集中outliers中具有较高hinge损失的样本索引
# 训练数据中的异常值，导致SVM分类错误的样本
inter_anomalies = list(set(train_outliers_index) & set(anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，但正确分类的预测值与剩余错误分类的最大预测值相差不足阈值1
inter_soft_anomalies = list(set(train_outliers_index) & set(soft_anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，且正确分类的预测值与剩余错误分类的最大预测值相差超过阈值1
inter_correct_class = list(set(train_outliers_index) & set(correct_class))

print("*" * 100)
print("训练集中SVM具有较高hinge损失函数的样本数量：", len(anomalies))
# print("训练集中SVM的hinge损失函数高于1的样本索引：", anomalies)
print("训练集中SVM分类错误的样本数量：", len(wrong_classified_indices))
# print("训练集中SVM分类错误的样本索引：", wrong_classified_indices)
intersection = np.intersect1d(anomalies, wrong_classified_indices)
diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("分类错误的样本中未被hinge阈值大于1识别的样本索引：", diff_elements)
print("hinge损失大于1的样本和分类错误的样本的交集数量：", len(intersection))
print("该交集所占分类错误的样本的比例：", len(intersection)/len(wrong_classified_indices))
print("*" * 100)
print("训练集的异常值中损失函数高于阈值1的样本索引：", inter_anomalies)
print("训练集的异常值中损失函数在0和阈值1之间的样本索引：", inter_soft_anomalies)
print("训练集的异常值中损失函数为0的样本索引：", inter_correct_class)

# 对训练数据进行处理后重新训练SVM模型

# SECTION 原始数据中的svm分类准确度
print("*" * 100)
print("原始训练集中SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("原始测试集中SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1，且被判定为异常值的样本，重新在处理后的训练数据上训练SVM
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_anomalies] = False
# 使用布尔索引删除那些既被判定为异常值，又使得hinge损失高于1的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练SVM模型
svm_model_split = svm.SVC()
svm_model_split.fit(X_train_split, y_train_split)
print("*" * 100)
print("去除同时满足异常和损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_split, svm_model_split.predict(X_train_split))))
print("去除同时满足异常和损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_split.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中hinge损失函数高于1的样本，重新在处理后的训练数据上训练SVM
mask_h = np.ones(len(X_train), dtype=bool)
mask_h[anomalies] = False
X_train_h = X_train[mask_h]
y_train_h = y_train[mask_h]
svm_model_h = svm.SVC()
svm_model_h.fit(X_train_h, y_train_h)
print("*" * 100)
print("去除损失高于阈值的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_h, svm_model_h.predict(X_train_h))))
print("去除损失高于阈值的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_h.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉SVM训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练SVM
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
svm_model_o = svm.SVC()
svm_model_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除判定为异常的样本后的训练集SVM分类准确度：" + str(accuracy_score(y_train_o, svm_model_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model_o.predict(X_test))))
print("*" * 100)