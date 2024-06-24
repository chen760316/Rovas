"""
在scikit-learn中，一共有3个朴素贝叶斯的分类算法
1、GaussianNB：GaussianNB就是先验为高斯分布（正态分布）的朴素贝叶斯，假设每个标签的数据都服从简单的正态分布
2、MultinomialNB：MultinomialNB就是先验为多项式分布的朴素贝叶斯。它假设特征是由一个简单多项式分布生成的
3、BernoulliNB：BernoulliNB就是先验为伯努利分布的朴素贝叶斯，在文本分类中，就是一个特征有没有在一个文档中出现
总结：
如果样本特征的分布大部分是连续值，使用GaussianNB会比较好；
如果样本特征的分布大部分是多元离散值，使用MultinomialNB比较合适(常用于文本分类)；
如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB
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
from sklearn.naive_bayes import GaussianNB

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

# SECTION 朴素贝叶斯分类器训练和分类准确度
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, y_train)
wrong_classified_indices = np.where(y_train != bayes_clf.predict(X_train))[0]
X_train_outliers = X_train[train_outliers_index]
y_train_outliers_numpy = y_train[train_outliers_index]
y_train_outliers = pd.Series(y_train_outliers_numpy)
print("*" * 100)
print("原始训练集中贝叶斯分类准确度：" + str(accuracy_score(y_train, bayes_clf.predict(X_train))))
print("原始测试集中贝叶斯分类准确度：" + str(accuracy_score(y_test, bayes_clf.predict(X_test))))
if not y_train_outliers.empty:
    print("训练集中异常值的贝叶斯分类准确度：" + str(accuracy_score(y_train_outliers, bayes_clf.predict(X_train_outliers))))
print("*" * 100)

# 训练集中分类正确的样本下标索引
corr_indices = np.where(y_train == bayes_clf.predict(X_train))[0]
# 训练集中分类错误的样本下标索引
wrong_indices = np.where(y_train != bayes_clf.predict(X_train))[0]
# 训练集中异常值中分类正确的样本下标索引
common_indices = np.where(y_train_outliers == bayes_clf.predict(X_train_outliers))[0]
# 训练集中异常值中分类错误的样本下标索引
diff_indices = np.where(y_train_outliers != bayes_clf.predict(X_train_outliers))[0]

# SECTION 朴素贝叶斯模型评价
from sklearn.metrics import classification_report, confusion_matrix
# 混淆矩阵，横轴为类别索引，纵轴为预测是否属于该类别，属于的话标明支持样本的数量
print(confusion_matrix(y_test, bayes_clf.predict(X_test)))
# macro avg：宏平均， 对指标所属各个类别的值直接取平均
# weighted avg ：加权平均，结合各个类别的数量加权取平均
print(classification_report(y_test, bayes_clf.predict(X_test)))

# section 朴素贝叶斯是基于概率生成的方法，并不涉及损失函数的优化,对于朴素贝叶斯模型，更适合的评价指标如准确率、召回率等。

# SECTION 原始数据中的朴素贝叶斯的分类准确度
print("*" * 100)
print("原始训练集中朴素贝叶斯的分类准确度：" + str(accuracy_score(y_train, bayes_clf.predict(X_train))))
print("原始测试集中朴素贝叶斯的分类准确度：" + str(accuracy_score(y_test, bayes_clf.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉朴素贝叶斯中分类错误的，且被判定为异常值的样本，重新在处理后的训练数据上训练朴素贝叶斯
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[diff_indices] = False
# 使用布尔索引删除那些既被判定为异常值，又被朴素贝叶斯分类错误的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练朴素贝叶斯模型
bayes_split = GaussianNB()
bayes_split.fit(X_train_split, y_train_split)
print("*" * 100)
print("去除朴素贝叶斯分类错误且被判定为异常的训练样本后，在训练集上朴素贝叶斯的分类准确度：" + str(accuracy_score(y_train_split, bayes_split.predict(X_train_split))))
print("去除朴素贝叶斯分类错误且被判定为异常的训练样本后，在测试集上朴素贝叶斯的分类准确度：" + str(accuracy_score(y_test, bayes_split.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉朴素贝叶斯训练数据中分类错误的训练样本，重新在处理后的训练数据上训练朴素贝叶斯
mask_cross = np.ones(len(X_train), dtype=bool)
mask_cross[wrong_indices] = False
X_train_cross = X_train[mask_cross]
y_train_cross = y_train[mask_cross]
bayes_cross = GaussianNB()
bayes_cross.fit(X_train_cross, y_train_cross)
print("*" * 100)
print("去除朴素贝叶斯训练数据中分类错误的训练样本，在训练集上朴素贝叶斯的分类准确度：" + str(accuracy_score(y_train_cross, bayes_cross.predict(X_train_cross))))
print("去除朴素贝叶斯训练数据中分类错误的训练样本，在测试集上朴素贝叶斯的分类准确度：" + str(accuracy_score(y_test, bayes_cross.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉朴素贝叶斯训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练朴素贝叶斯
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
bayes_o = GaussianNB()
bayes_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除朴素贝叶斯训练数据中被判定为异常的样本后，在训练集朴素贝叶斯的分类准确度：" + str(accuracy_score(y_train_o, bayes_o.predict(X_train_o))))
print("去除朴素贝叶斯训练数据中被判定为异常的样本后，在测试集朴素贝叶斯的分类准确度：" + str(accuracy_score(y_test, bayes_o.predict(X_test))))
print("*" * 100)

# section 计算训练样本的迭代损失，朴素贝叶斯不涉及显示的训练过程，没有迭代损失这一概念