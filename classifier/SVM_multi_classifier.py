"""
1、支持向量机（SVM）中，训练过程通常是基于一对多（One-vs-Rest，OvR）策略，尤其是在多类别分类问题中。这意味着对于有 num_classes 个类别的问题，训练会涉及训练 num_classes 个分类器，每个分类器用来区分一个类别与其他所有类别的组合
2、OvR 训练策略：对于每个类别 j，一个 SVM 分类器会被训练来将该类别作为正例，而将所有其他类别作为负例。这个分类器会学习一个超平面，以区分其对应的类别和所有其他类别之间的差异
3、decision_function 输出：在测试阶段，svm_model.decision_function(X_test) 会为测试集中的每个样本提供一个分数向量，其中每个分数表示该样本与每个超平面之间的距离或置信度。
通常情况下，分数向量的维度是 (num_samples, num_classes)，每个元素 decision_values[i, j] 表示第 i 个样本属于第 j 类的置信度。决策函数 decision_function 的输出并不是类别的概率预测值，而是一个衡量样本到决策边界距离的数值
4、样本的hinge损失大于1意味着被误分类
5、SVM训练目标为找到每个类别的分类超平面，通常不涉及多轮迭代训练损失函数
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

"""
kaggle datasets
数据预处理
"""

"""dry_bean数据集"""
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
drybean = pd.read_excel(file_path)
enc = LabelEncoder()
drybean['Class'] = enc.fit_transform(drybean['Class'])
X = drybean.values[:,0:16]
y = drybean.values[:,16]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
test_data = pd.concat([X_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
test_data.columns = drybean.columns.values

"""obesity数据集"""
# file_path = "../kaggle_datasets/Obesity_prediction/obesity_data.csv"
# label_col_name = "ObesityCategory"
# data = pd.read_csv(file_path)
# enc = LabelEncoder()
# data[label_col_name] = enc.fit_transform(data[label_col_name])
# categorical_features = [1]
# categorical_names = {}
# feature_names = data.columns[:-1].tolist()
# # 对字符串列进行数值映射
# for feature in categorical_features:
#     le = LabelEncoder()
#     le.fit(data.iloc[:, feature])
#     data.iloc[:, feature] = le.transform(data.iloc[:, feature])
#     categorical_names[feature] = le.classes_
# data[feature_names] = data[feature_names].astype(float)
# num_col = data.shape[1]
# X = data.values[:, 0:num_col-1]
# y = data.values[:, num_col-1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

"""wine-quality数据集"""
# file_path = "../UCI_datasets/wine+quality/winequality-white.csv"
# label_col_name = "quality"
# data = pd.read_csv(file_path, sep=';')
# enc = LabelEncoder()
# data[label_col_name] = enc.fit_transform(data[label_col_name])
# feature_names = data.columns[:-1].tolist()
# data[feature_names] = data[feature_names].astype(float)
# data[label_col_name] = data[label_col_name].astype(int)
# num_col = data.shape[1]
# X = data.values[:, 0:num_col-1]
# y = data.values[:, num_col-1]
# # 对不同维度进行标准化
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

"""
GOAD异常检测器
"""
clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf.fit(X_train, y=None)

"""
DeepSAD异常检测器
"""
# clf = DeepSAD(epochs=1, hidden_dims=20,
#                    device=device,
#                    random_state=42)
# anom_id = np.where(y_train == 1)[0]
# known_anom_id = np.random.choice(anom_id, 10, replace=False)
# y_semi = np.zeros_like(y_train, dtype=int)
# y_semi[known_anom_id] = 1
# clf.fit(X_train, y_semi)

scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)
outliers_index = []
test_class_num = 0

# 训练集上的分数
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

# # 测试集中指定标签列类别为target_class下的异常值检测
# for i in range(len(scores)):
#     # print(scores[i], pred_labels[i], confidence[i])
#     # if pred_labels[i] == 1:
#     #     outliers_index.append(i)
#     if y_test[i] == target_class:
#         test_class_num += 1
#         if pred_labels[i] == 1:
#             outliers_index.append(i)
# print("指定标签类下测试集中异常值索引：", outliers_index)
# print("指定标签类下测试集中的异常值数量：", len(outliers_index))
# print("指定标签类下测试集样本数：", test_class_num)
# if test_class_num > 0:
#     print("指定标签类下测试集中异常点所占比例为：", len(outliers_index)/test_class_num)

# 测试集中标签列所有类别下的的异常值检测
test_class_num = len(y_test)
print("测试集样本数：", test_class_num)
for i in range(len(scores)):
    if pred_labels[i] == 1:
        outliers_index.append(i)
print("测试集中异常值索引：", outliers_index)
print("测试集中的异常值数量：", len(outliers_index))

"""
SVM模型训练和分类准确度
"""
svm_model = svm.SVC()
# svm_model = svm.SVC(C=10)  # 默认使用 RBF 核函数（径向基函数），即高斯核函数
# svm_model = svm.SVC(C=2, gamma=0.1)
# svm_model = svm.SVC(kernel='linear')  # 线性核函数
# svm_model = svm.SVC(kernel='poly', degree=3, coef0=1)  # 多项式核函数
# svm_model = svm.SVC(kernel='sigmoid', gamma=0.1, coef0=0.5)  # sigmoid核函数

svm_model.fit(X_train, y_train)
# SVM分类准确度
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
test_label_pred = svm_model.predict(X_test)
wrong_classified_indices = np.where(y_test != svm_model.predict(X_test))[0]
X_test_outliers = X_test[outliers_index]
y_test_outliers_numpy = y_test[outliers_index]
y_test_outliers = pd.Series(y_test_outliers_numpy)
if not y_test_outliers.empty:
    print("测试集中异常值的SVM分类准确度：" + str(accuracy_score(y_test_outliers, svm_model.predict(X_test_outliers))))

"""
自定义每个样本的hinge损失函数值
"""
# # 获取测试集每个样本的决策函数值
# y_predicate = svm_model.predict(X_test)
# decision_values = svm_model.decision_function(X_test)
# # 计算 Hinge Loss
# sample_num = decision_values.shape[0]
# num_classes = decision_values.shape[1]
# hinge_l = np.zeros(sample_num)
# y_train_one_hot = one_hot_encode(y_test, num_classes)
# for i in range(sample_num):
#     true_class_indices = np.where(y_train_one_hot[i] == 1)[0].item()
#     for j in range(num_classes):
#         if y_train_one_hot[i, j] == 1:
#             continue
#         else:
#             hinge_l[i] += max(decision_values[i, j] - decision_values[i, true_class_indices] + 1, 0)
# print("每个样本的 Hinge Loss：")
# print(hinge_l[:100])
# # 判定异常：假设阈值为 1，超过此值即认为是异常
# anomalies = np.where(hinge_l > 1)[0]
# print("利用SVM损失函数阈值判别的异常样本索引：", anomalies)

"""
使用sklearn库中的hinge损失函数
"""
decision_values = svm_model.decision_function(X_test)
predicted_classes = np.argmax(decision_values, axis=1)
print("使用hinge损失预测标签的测试集SVM分类准确度：" + str(accuracy_score(y_test, predicted_classes)))
# 计算训练样本的平均损失
test_losses = hinge_loss(y_test, decision_values, labels=np.unique(y_test))
print("整个测试集下的平均hinge损失：", test_losses)
# 计算每个样本的hinge损失
num_samples = X_test.shape[0]
num_classes = svm_model.classes_.shape[0]
hinge_losses = np.zeros((num_samples, num_classes))
hinge_loss = np.zeros(num_samples)
for i in range(num_samples):
    correct_class = int(y_test[i])
    for j in range(num_classes):
        if j != correct_class:
            loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
            hinge_losses[i, j] = loss_j
    # hinge_loss[i] = np.sum(hinge_losses[i])
    hinge_loss[i] = np.max(hinge_losses[i])
# print("每个样本的各分类类别 Hinge Loss：", hinge_losses[:100])
# print("每个样本的总体 Hinge Loss：", hinge_loss[:100])
# 判定异常：假设阈值为 1，超过此值即认为是异常
anomalies = np.where(hinge_loss > 1)[0]
soft_anomalies = np.where((hinge_loss > 0) & (hinge_loss <= 1))[0]
correct_class = np.where(hinge_loss == 0)[0]
# 输出测试集中outliers中具有较高hinge损失的样本索引
# 测试数据中的异常值，导致SVM分类错误的样本
inter_anomalies = list(set(outliers_index) & set(anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，但正确分类的预测值与剩余错误分类的最大预测值相差不足阈值
inter_soft_anomalies = list(set(outliers_index) & set(soft_anomalies))
# 测试数据中的潜在异常值，未导致SVM分类错误，且正确分类的预测值与剩余错误分类的最大预测值相差超过阈值
inter_correct_class = list(set(outliers_index) & set(correct_class))

print("*" * 100)
# print("测试集中异常值索引为：", outliers_index)
# print("测试集中具有较高损失函数的样本索引：", anomalies)
# print("测试集中分类错误的样本索引：", wrong_classified_indices)
print("测试集中SVM具有较高hinge损失函数的样本数量：", len(anomalies))
print("测试集中SVM分类错误的样本数量：", len(wrong_classified_indices))
print("测试集中SVM分类错误的样本索引：", wrong_classified_indices)
intersection = np.intersect1d(anomalies, wrong_classified_indices)

diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("分类错误的样本中未被hinge阈值识别的样本索引：", diff_elements)
print("较高损失函数的样本和分类错误的样本的交集数量：", len(intersection))
print("较高损失函数的样本和分类错误的样本的交集，所占分类错误的样本的比例：", len(intersection)/len(wrong_classified_indices))
print("*" * 100)
print("测试集的异常值中损失函数高于阈值的样本索引：", inter_anomalies)
print("测试集的异常值中损失函数在0和阈值之间的样本索引：", inter_soft_anomalies)
print("测试集的异常值中损失函数为0的样本索引：", inter_correct_class)
