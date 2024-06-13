# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, NeuTraL
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

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
target_class = 6
"""
kaggle datasets
"""
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
drybean = pd.read_excel(file_path)
enc = LabelEncoder()
drybean['Class'] = enc.fit_transform(drybean['Class'])
X = drybean.values[:,0:16]
y = drybean.values[:,16]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
"""GOAD异常检测器"""
clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
pred_labels, confidence = clf.predict(X_test, return_confidence=True)
print("训练集中异常值判定阈值为：", clf.threshold_)

outliers_index = []
test_class_num = 0
for i in range(len(scores)):
    # print(scores[i], pred_labels[i], confidence[i])
    # if pred_labels[i] == 1:
    #     outliers_index.append(i)
    if y_test[i] == target_class:
        test_class_num += 1
        if pred_labels[i] == 1:
            outliers_index.append(i)
print("异常检测器检测出的测试集中异常值索引为：", outliers_index)
print("测试集中指定类异常点数：",len(outliers_index))
print("测试集中指定类点数：",test_class_num)
print("测试集中指定类异常点所占比例为：", len(outliers_index)/test_class_num)
# print("测试集中异常点所占比例为：", len(outliers_index)/len(scores))
svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)

# 获取测试集每个样本的决策函数值
y_predicate = svm_model.predict(X_test)
decision_values = svm_model.decision_function(X_test)
# 计算 Hinge Loss
sample_num = decision_values.shape[0]
num_classes = decision_values.shape[1]
hinge_loss = np.zeros(sample_num)
y_train_one_hot = one_hot_encode(y_test, num_classes)
for i in range(sample_num):
    true_class_indices = np.where(y_train_one_hot[i] == 1)[0].item()
    for j in range(num_classes):
        if y_train_one_hot[i, j] == 1:
            continue
        else:
            hinge_loss[i] += max(decision_values[i, j] - decision_values[i, true_class_indices] + 1, 0)
# 判定异常：假设阈值为 1，超过此值即认为是异常
anomalies = np.where(hinge_loss > 1)[0]
print("利用SVM损失函数阈值判别的异常样本索引：", anomalies)

# SVM分类准确度
print("训练集SVM分类准确度：" + str(accuracy_score(y_train, svm_model.predict(X_train))))
print("测试集SVM分类准确度：" + str(accuracy_score(y_test, svm_model.predict(X_test))))
X_test_outliers = X_test[outliers_index]
y_test_outliers_numpy = y_test[outliers_index]
y_test_outliers = pd.Series(y_test_outliers_numpy)
print("测试集中指定类异常点的SVM分类准确度：" + str(accuracy_score(y_test_outliers, svm_model.predict(X_test_outliers))))