"""
机器学习谓词第二种：重要特征检测相关方法
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_friedman1

"""
在实际应用中，常见的特征选择方法包括基于统计特征重要性的方法（如方差分析、卡方检验、信息增益等）、
基于模型性能的方法（如基于树模型的特征重要性、基于正则化的特征选择方法）等
"""

# 生成示例数据
np.random.seed(0)
num_samples = 1000
num_features = 10
num_labels = 5

# 生成特征数据
X = np.random.rand(num_samples, num_features)

# 生成随机的标签数据
y = np.random.randint(0, num_labels, num_samples).reshape(-1, 1)

# 拼接为完整数据
data = np.concatenate((X, y), axis=1)

"""借助方差筛选重要特征(特征的自分布信息)"""
top_k_var = 3
variances = np.var(X, axis=0)
top_k_indices_var = np.argsort(variances)[-top_k_var:][::-1]
top_k_variances_var = variances[top_k_indices_var]
print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices_var))
print("对应的方差数值：", top_k_variances_var)

"""借助pearson相关系数筛选重要特征(单个特征和标签之间的线性相关性)"""
top_k_pearson = 3
y_trans = y.reshape(-1)
pearson_matrix = np.corrcoef(X.T, y_trans)
correlations = np.abs(pearson_matrix[0, 1:])
top_k_indices_pearson = np.argsort(correlations)[::-1][:top_k_pearson]
top_k_correlations_pearson = correlations[top_k_indices_pearson]
print("与标签y的Pearson相关系数绝对值最大的前{}个特征的索引：{}".format(top_k_pearson, top_k_indices_pearson))
print("对应的相关系数绝对值：", top_k_correlations_pearson)

"""借助互信息筛选重要特征(单个特征和标签之间的互信息)"""
top_k_mi = 3
y_trans_mi = y.reshape(-1)
mi = mutual_info_regression(X, y_trans_mi)
top_k_indices = np.argsort(mi)[::-1][:top_k_mi]
top_k_coef = mi[top_k_indices]
print("互信息最多的前{}个特征的索引：{}".format(top_k_mi, top_k_indices))
print("对应的互信息：", top_k_coef)

"""借助lasso筛选重要特征(特征的联合分布和标签之间的线性相关性)"""
alpha = 0.0001
top_k_lasso = 3
lasso = Lasso(alpha)
lasso.fit(X, y)
coef = lasso.coef_
coef_abs = abs(coef)
top_k_indices = np.argsort(coef_abs)[::-1][:top_k_lasso]
top_k_coef = coef[top_k_indices]
print("绝对值最大的前{}个属性的索引：{}".format(top_k_lasso, top_k_indices))
print("对应的系数：", top_k_coef)

"""借助Fisher检验筛选最有影响力的k个特征"""
top_k_fisher = 3
selector = SelectKBest(score_func=f_classif, k=top_k_fisher)
y_trans_fisher = y.reshape(-1)
X_new = selector.fit_transform(X, y_trans_fisher)
fisher_scores = selector.scores_
# 获取被选中的特征的索引
selected_feature_indices = selector.get_support(indices=True)
print("最有影响力的{}个特征的索引：{}".format(top_k_fisher, selected_feature_indices))
print("每个特征的Fisher检验得分：", fisher_scores)

"""借助CART决策树筛选最有影响力的k个特征"""
top_k_cart = 3
clf = DecisionTreeClassifier()
clf.fit(X, y)
# 获取特征重要性得分
feature_importance = clf.feature_importances_
# 根据重要性得分降序排序
sorted_indices = np.argsort(feature_importance)[::-1]
# 根据重要性得分降序排序
top_k_features = sorted_indices[:top_k_cart]
print("最有影响力的的前{}个属性的索引：{}".format(top_k_cart, top_k_features))
print("特征重要性得分为：{}".format(feature_importance))

"""sklearn库筛选重要特征的方法"""
# 通过使用SelectFromModel或SelectKBest，你可以根据不同的评估器或评分函数选择与目标变量相关的最重要的特征，选择哪种方法取决于你的数据和具体的需求
"""
case1: SelectFromModel选择器
SelectFromModel：SelectFromModel是一种元变换器（transformer），
它可以与任何具有coef_ 或 feature_importances_ 属性（如随机森林和决策树模型）的评估器一起使用来选择特征
"""
# 加载鸢尾花数据集作为示例
data = load_iris()
X, y = data.data, data.target
# 实例化随机森林分类器作为特征选择器
clf = RandomForestClassifier()
clf.fit(X, y)
# 使用SelectFromModel来选择重要特征
sfm = SelectFromModel(clf, threshold='mean', prefit=True)
X_selected = sfm.transform(X)
# 获取选择的特征索引
selected_idx = sfm.get_support(indices=True)
# 打印选择的特征索引
print("选择的特征索引:", selected_idx)
# 打印特征重要性
feature_importance = clf.feature_importances_
print("特征重要性:", feature_importance)

"""
case2：SelectKBest选择器
SelectKBest是另一种方法，它根据指定的统计测试（比如卡方、F检验等）来选择前k个最好的特征
"""
# 假设X和Y是你的特征和目标变量
X, Y = make_classification(n_samples=100, n_features=20, random_state=0)
# 使用SelectKBest来选择k个最好的特征
k = 5  # 假设选择前5个特征
skb = SelectKBest(score_func=f_classif, k=k)
X_selected = skb.fit_transform(X, Y)
# 获取选择的特征索引
selected_idx = skb.get_support(indices=True)
# 打印选择的特征索引
print("选择的特征索引:", selected_idx)
# 打印特征评分
feature_scores = skb.scores_
print("特征评分:", feature_scores)

"""利用VC维去除无效特征(伪代码)"""
# X, Y = make_classification(n_samples=500, n_features=20, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# # 初始化支持向量机分类器模型
# model = SVC()
# # 计算原始特征集合的模型性能
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# original_accuracy = accuracy_score(y_test, y_pred)
# # 计算每个特征的VC维(伪代码，具体实现很复杂)
# vc_dimension = calculate_vc_dimension(X_train, y_train)
# # 根据VC维去除无效特征(伪代码)
# selected_features = select_features_based_on_vc_dimension(X_train, vc_dimension)
# # 基于去除无效特征后的特征集合训练模型并计算性能
# model.fit(X_train[:, selected_features], y_train)
# y_pred = model.predict(X_test[:, selected_features])
# selected_accuracy = accuracy_score(y_test, y_pred)
# # 输出结果
# print("原始特征集合的准确率:", original_accuracy)
# print("去除无效特征后的准确率:", selected_accuracy)

"""借助wrapper(包装)方法生成特征子集"""
# 生成示例数据
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
# 初始化线性回归模型
model = LinearRegression()
# 初始化 RFE 特征选择器，选择要保留的特征数量
rfe = RFE(model, n_features_to_select=5)
# 拟合 RFE 特征选择器
rfe.fit(X, y)
# 输出选择的特征
print("Selected features:", rfe.support_)
# 输出特征排名
print("Feature ranking:", rfe.ranking_)

