"""
1、ID3算法：其核心是在决策树的各级节点上，使用信息增益方法的选择标准，来帮助确定生产每个节点时所对应采用的合适属性，不能自动分箱，不能剪枝；
2、C4.5算法：相对于ID3改进是使用信息增益率来选择节点属性。克服ID3点不足： ID3只适用于离散的描述属性，C4.5可以处理连续和离散属性；可以剪枝；
3、CART算法：通过构建树、修剪树、评估树来构建一个二叉树。通过控制树的结构来控制模型。当终节点是连续变量是——回归树，当终节点是分类变量是——分类树；
4、Scikit-learn中的决策树模型（例如DecisionTreeClassifier）默认使用基尼系数（Gini impurity）作为分裂标准,
决策树在训练时并不直接优化损失函数，而是根据给定的分裂标准递归地构建树，直到达到停止条件为止;
5、决策树在训练时是通过递归地选择最佳的特征和切分点来构建树的节点，不涉及典型意义上的迭代更新参数。
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

import os
os.environ["PATH"] += os.pathsep + 'E:/graphviz/bin/'

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

# SECTION Scikit-learn中的决策树模型（例如DecisionTreeClassifier）默认使用基尼系数（Gini impurity）作为分裂标准,
#  决策树在训练时并不直接优化损失函数，而是根据给定的分裂标准递归地构建树，直到达到停止条件为止
from sklearn.metrics import classification_report, confusion_matrix
cart_classifier = DecisionTreeClassifier()
# cart_classifier = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
# cart_classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=20)
cart_classifier.fit(X_train, y_train)
y_pred = cart_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
wrong_classified_indices = np.where(y_train != cart_classifier.predict(X_train))[0]
X_train_outliers = X_train[train_outliers_index]
y_train_outliers_numpy = y_train[train_outliers_index]
y_train_outliers = pd.Series(y_train_outliers_numpy)
print("*" * 100)
print("原始训练集中CART决策树分类准确度：" + str(accuracy_score(y_train, cart_classifier.predict(X_train))))
print("原始测试集中CART决策树分类准确度：" + str(accuracy_score(y_test, cart_classifier.predict(X_test))))
if not y_train_outliers.empty:
    print("训练集中异常值的CART决策树分类准确度：" + str(accuracy_score(y_train_outliers, cart_classifier.predict(X_train_outliers))))
print("*" * 100)
# 训练集中分类正确的样本下标索引
corr_indices = np.where(y_train == cart_classifier.predict(X_train))[0]
# 训练集中分类错误的样本下标索引
wrong_indices = np.where(y_train != cart_classifier.predict(X_train))[0]
# 训练集中异常值中分类正确的样本下标索引
common_indices = np.where(y_train_outliers == cart_classifier.predict(X_train_outliers))[0]
# 训练集中异常值中分类错误的样本下标索引
diff_indices = np.where(y_train_outliers != cart_classifier.predict(X_train_outliers))[0]

# section 决策树可视化
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

# 将决策树导出为DOT格式
dot_data = StringIO()
export_graphviz(cart_classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

# 使用pydotplus生成决策树图形
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.set_size('"8,8!"')
graph.set_dpi(300)
# 可视化决策树
graph.write_svg("decision_tree.svg")
print("决策树图形已生成并保存为 decision_tree.svg")

# SECTION 原始数据中的决策树分类准确度
print("*" * 100)
print("原始训练集中决策树的分类准确度：" + str(accuracy_score(y_train, cart_classifier.predict(X_train))))
print("原始测试集中决策树的分类准确度：" + str(accuracy_score(y_test, cart_classifier.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉决策树中分类错误的，且被判定为异常值的样本，重新在处理后的训练数据上训练决策树
# 生成布尔索引，为要删除的行创建布尔值数组
mask = np.ones(len(X_train), dtype=bool)
mask[diff_indices] = False
# 使用布尔索引删除那些既被判定为异常值，又被决策树分类错误的样本
X_train_split = X_train[mask]
y_train_split = y_train[mask]
# 重新训练决策树模型
cart_split = DecisionTreeClassifier()
cart_split.fit(X_train_split, y_train_split)
print("*" * 100)
print("去除决策树分类错误且被判定为异常的训练样本后，在训练集上决策树的分类准确度：" + str(accuracy_score(y_train_split, cart_split.predict(X_train_split))))
print("去除决策树分类错误且被判定为异常的训练样本后，在测试集上决策树的分类准确度：" + str(accuracy_score(y_test, cart_split.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉决策树训练数据中分类错误的训练样本，重新在处理后的训练数据上训练决策树
mask_cross = np.ones(len(X_train), dtype=bool)
mask_cross[wrong_indices] = False
X_train_cross = X_train[mask_cross]
y_train_cross = y_train[mask_cross]
cart_cross = DecisionTreeClassifier()
cart_cross.fit(X_train_cross, y_train_cross)
print("*" * 100)
print("去除决策树训练数据中分类错误的训练样本，在训练集上决策树的分类准确度：" + str(accuracy_score(y_train_cross, cart_cross.predict(X_train_cross))))
print("去除决策树训练数据中分类错误的训练样本，在测试集上决策树的分类准确度：" + str(accuracy_score(y_test, cart_cross.predict(X_test))))
print("*" * 100)

# SECTION 舍弃掉决策树训练数据中被判定为异常值的样本，重新在处理后的训练数据上训练决策树
mask_o = np.ones(len(X_train), dtype=bool)
mask_o[train_outliers_index] = False
X_train_o = X_train[mask_o]
y_train_o = y_train[mask_o]
cart_o = DecisionTreeClassifier()
cart_o.fit(X_train_o, y_train_o)
print("*" * 100)
print("去除决策树训练数据中被判定为异常的样本后，在训练集决策树的分类准确度：" + str(accuracy_score(y_train_o, cart_o.predict(X_train_o))))
print("去除决策树训练数据中被判定为异常的样本后，在测试集决策树的分类准确度：" + str(accuracy_score(y_test, cart_o.predict(X_test))))
print("*" * 100)

# section 计算训练样本的迭代损失，决策树不涉及显示的训练过程，没有迭代损失这一概念