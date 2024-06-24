"""
使用softmax训练集求解交叉熵损失函数，因为测试集标签不可见，Softmax分类器可以看作是逻辑回归在多类别情况下的推广或扩展
softmax分类器和异常检测器使用相同的训练集和测试集
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
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
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

# SECTION softmax分类器训练和分类准确度
# 初始化 LogisticRegression 模型，用于多标签分类
softmax_classifier = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
softmax_classifier.fit(X_train, y_train)
wrong_classified_indices = np.where(y_train != softmax_classifier.predict(X_train))[0]
X_train_outliers = X_train[train_outliers_index]
y_train_outliers_numpy = y_train[train_outliers_index]
y_train_outliers = pd.Series(y_train_outliers_numpy)
print("*" * 100)
print("原始训练集中softmax分类准确度：" + str(accuracy_score(y_train, softmax_classifier.predict(X_train))))
print("原始测试集中softmax分类准确度：" + str(accuracy_score(y_test, softmax_classifier.predict(X_test))))
if not y_train_outliers.empty:
    print("训练集中异常值的softmax分类准确度：" + str(accuracy_score(y_train_outliers, softmax_classifier.predict(X_train_outliers))))
print("*" * 100)

# SECTION 使用sklearn库中的交叉熵损失函数
# 获取训练集样本属于每一类的目标函数值
decision_values = softmax_classifier.decision_function(X_train)
# 预测每个样本被划分到的类别
predicted_labels = softmax_classifier.predict(X_train)
# 预测每个样本被划分到每个类别的概率
probabilities = softmax_classifier.predict_proba(X_train)
# 计算交叉熵损失（平均损失）
average_loss = log_loss(y_train, probabilities)
print("整个训练集下的交叉熵损失为：", average_loss)

# section 计算每个样本的交叉熵损失
# 输出每个训练样本的交叉熵损失
cross_entropy_loss = []
for i in range(len(X_train)):
    y_true_index = int(y_train[i])
    y_true = np.zeros(len(softmax_classifier.classes_))
    y_true[y_true_index] = 1
    sample_loss = -np.sum(y_true * np.log(probabilities[i]))
    cross_entropy_loss.append(sample_loss)
# 通过交叉熵损失判定异常：假设阈值为 1，超过此值即认为是异常
# 计算损失的平均值和标准差
mean_loss = np.mean(cross_entropy_loss)
std_loss = np.std(cross_entropy_loss)
# 设置损失阈值为平均值加上两倍标准差
threshold = mean_loss + 2 * std_loss
anomalies = np.where(cross_entropy_loss >= threshold)[0]
correct_samples = np.where(cross_entropy_loss <= threshold)[0]
# 输出训练集中outliers中具有较高交叉熵损失的样本索引
# 训练数据中的异常值中交叉熵损失较大，对softmax分类器有害的样本
inter_anomalies = list(set(train_outliers_index) & set(anomalies))
# 训练数据中的异常值中交叉熵损失较小的样本
inter_correct_class = list(set(train_outliers_index) & set(correct_samples))

print("*" * 100)
print("训练集中具有较高交叉熵损失的样本数量：", len(anomalies))
print("训练集中softmax分类错误的样本数量：", len(wrong_classified_indices))
intersection = np.intersect1d(anomalies, wrong_classified_indices)
diff_elements = np.setdiff1d(wrong_classified_indices, intersection)
print("分类错误的样本中交叉熵损失较小的的样本索引：", diff_elements)
print("交叉熵损失较大的样本和分类错误的样本的交集数量：", len(intersection))
print("该交集所占分类错误的样本的比例：", len(intersection)/len(wrong_classified_indices))
print("*" * 100)
print("训练集的异常值中交叉熵损失较高的样本索引：", inter_anomalies)
print("训练集的异常值中交叉熵损失较低的的样本索引：", inter_correct_class)

# SECTION 原始数据中的softmax分类准确度
print("*" * 100)
print("原始训练集中softmax分类准确度：" + str(accuracy_score(y_train, softmax_classifier.predict(X_train))))
print("原始测试集中softmax分类准确度：" + str(accuracy_score(y_test, softmax_classifier.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉softmax训练数据中交叉熵损失较高，且被判定为异常值的样本，重新在处理后的训练数据上训练softmax分类器
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[inter_anomalies] = False
# 使用布尔索引删除那些既被判定为异常值，又具有较高交叉熵损失值的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练softmax模型
softmax_classifier_split = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
softmax_classifier_split.fit(X_train_split, y_train_split)
print("*" * 100)
print("去除同时满足异常和损失高于阈值的样本后的训练集softmax分类准确度：" + str(accuracy_score(y_train_split, softmax_classifier_split.predict(X_train_split))))
print("去除同时满足异常和损失高于阈值的样本后的测试集softmax分类准确度：" + str(accuracy_score(y_test, softmax_classifier_split.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉softmax分类器中交叉熵损失高于阈值的样本，重新在处理后的训练数据上训练softmax分类器
mask_cross = np.ones(len(X_train), dtype=bool)
mask_cross[anomalies] = False
X_train_cross = X_train[mask_cross]
y_train_cross = y_train[mask_cross]
softmax_classifier_cross = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
softmax_classifier_cross.fit(X_train_cross, y_train_cross)
print("*" * 100)
print("去除损失高于阈值的样本后的训练集softmax分类准确度：" + str(accuracy_score(y_train_cross, softmax_classifier_cross.predict(X_train_cross))))
print("去除损失高于阈值的样本后的测试集softmax分类准确度：" + str(accuracy_score(y_test, softmax_classifier_cross.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉softmax训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练softmax分类器
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
softmax_classifier_o = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
softmax_classifier_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除判定为异常的样本后的训练集softmax分类准确度：" + str(accuracy_score(y_train_o, softmax_classifier_o.predict(X_train_o))))
print("去除判定为异常的样本后的测试集softmax分类准确度：" + str(accuracy_score(y_test, softmax_classifier_o.predict(X_test))))
print("*" * 100)

# section 计算训练样本的迭代损失
# 训练过程中记录每轮迭代的损失值
losses_per_iteration = []
# 开始训练迭代
interval = 1
# outlier_out_threshold为被异常检测器判定为异常值，且交叉熵损失高于阈值的样本
outlier_out_threshold = inter_anomalies
for iteration in range(10):  # 假设迭代10轮
    softmax_classifier_iteration = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, max_iter=interval*(iteration+1))
    # 在每轮迭代中训练 softmax 分类器
    softmax_classifier_iteration.fit(X_train, y_train)
    # 获取决策函数值
    decision_values = softmax_classifier_iteration.decision_function(X_train)
    num_classes_iteration = softmax_classifier_iteration.classes_.shape[0]
    # 预测每个样本被划分到每个类别的概率
    probabilities = softmax_classifier_iteration.predict_proba(X_train)
    # 计算每轮迭代的平均交叉熵损失
    average_loss = log_loss(y_train, probabilities)
    # 将每轮迭代的损失值记录下来
    losses_per_iteration.append(average_loss)
    print("-"*100)
    print(f"Iteration {(iteration + 1) * interval}")
    for i in outlier_out_threshold:
        y_true_index = int(y_train[i])
        y_true = np.zeros(len(softmax_classifier_iteration.classes_))
        y_true[y_true_index] = 1
        sample_loss = -np.sum(y_true * np.log(probabilities[i]))
        print(f"Sample {i} has cross entropy losses = {sample_loss}")
# 输出每轮迭代的损失值
print("*" * 100)
for i, losses in enumerate(losses_per_iteration):
    print(f"Iteration {(i + 1)*interval}: Mean cross entropy loss = {np.mean(losses)}")
print("*" * 100)