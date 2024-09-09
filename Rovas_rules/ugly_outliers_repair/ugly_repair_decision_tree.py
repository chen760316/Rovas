"""
方案一效果好，方案二效果有限
采用决策树修改异常值的标签
采用统计方法修复异常值的特征
使用decision_tree分类器进行实验
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理
file_path = "../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
# 原始数据集D对应的Dataframe
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.2, random_state=1)
# 加入随机噪声的比例
noise_level = 0.2
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)

# section 向数据集中加噪
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从含噪数据中生成训练数据和测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
all_columns = data.columns.values.tolist()
feature_names = all_columns[:-1]
class_name = all_columns[-1]
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=all_columns)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# section 找到有影响力的特征 M𝑐 (𝑅, 𝐴, M)
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)

i = 16
np.random.seed(1)
categorical_features = [0, 6]
categorical_names = {}

# 设置决策树参数
params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid_search.fit(X_train_copy, y_train)

# 获取最佳模型
decision_tree_classifier = grid_search.best_estimator_

for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data_copy.iloc[:, feature])
    data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
    categorical_names[feature] = le.classes_

explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

predict_fn = lambda x: decision_tree_classifier.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=6)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
important_features = []
for feature_set in top_features:
    feature_long = feature_set[0]
    for feature in feature_names:
        if set(feature).issubset(set(feature_long)):
            important_features.append(feature)
            break

top_k_indices = [feature_names.index(feature_name) for feature_name in important_features]
print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# section 找到loss(M, D, 𝑡) > 𝜆的元组

# choice 计算分类错误的样本

# 预测每个样本被划分到的类别
predicted_labels = decision_tree_classifier.predict(X_copy)
ugly_outlier_candidates = np.where(y != predicted_labels)[0]

# section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现，找到所有有影响力的特征下的异常元组

outlier_feature_indices = {}
threshold = 0.01
for column_indice in top_k_indices:
    select_feature = feature_names[column_indice]
    select_column_data = data_copy[select_feature].values
    max_value = np.max(select_column_data)
    min_value = np.min(select_column_data)
    sorted_indices = np.argsort(select_column_data)
    sorted_data = select_column_data[sorted_indices]
    # 找到A属性下的所有异常值
    outliers = []
    outliers_index = []
    # 检查列表首尾元素
    if len(sorted_data) > 1:
        if (sorted_data[1] - sorted_data[0] >= threshold):
            outliers.append(sorted_data[0])
            outliers_index.append(sorted_indices[0])
        if (sorted_data[-1] - sorted_data[-2] >= threshold):
            outliers.append(sorted_data[-1])
            outliers_index.append(sorted_indices[-1])
    # 检查中间元素
    for i in range(1, len(sorted_data) - 1):
        current_value = sorted_data[i]
        left_value = sorted_data[i - 1]
        right_value = sorted_data[i + 1]
        if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
            outliers.append(current_value)
            outliers_index.append(sorted_indices[i])
    outliers_index_numpy = np.array(outliers_index)
    intersection = np.intersect1d(np.array(outliers_index), ugly_outlier_candidates)
    # print("有影响力的特征A下同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆的所有异常值索引为：", intersection)
    outlier_feature_indices[column_indice] = intersection
# print(outlier_feature_indices)

# SECTION decision_tree模型的实现

# subsection 原始数据集上训练的decision_tree模型在训练集和测试集中分错的样本比例

print("*" * 100)
# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 获取最佳模型
decision_tree_clf = grid_search.best_estimator_
train_label_pred = decision_tree_clf.predict(X_train)
test_label_pred = decision_tree_clf.predict(X_test)

# 训练样本中被decision_tree模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
print("训练样本中被decision_tree模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被decision_tree模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
print("测试样本中被decision_tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被decision_tree模型错误分类的样本
print("完整数据集D中被decision_tree模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的decision_tree模型在训练集和测试集中分错的样本比例

print("*" * 100)
train_label_pred_noise = decision_tree_classifier.predict(X_train_copy)
test_label_pred_noise = decision_tree_classifier.predict(X_test_copy)

# 加噪训练样本中被decision_tree模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
print("加噪训练样本中被decision_tree模型错误分类的样本占总加噪训练样本的比例：",
      len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被decision_tree模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
print("加噪测试样本中被decision_tree模型错误分类的样本占总测试样本的比例：",
      len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被decision_tree模型错误分类的样本
print("完整数据集D中被decision_tree模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 方案一：对X_copy中需要修复的元组进行标签修复（decision_tree方法）
#  需要修复的元组通过异常值检测器检测到的元组和decision_tree分类错误的元组共同确定（取并集）

# subsection 尝试修复异常数据的标签

# 确定数据集D中需要修复的元组和正常元组
outlier_tuple_set = set()
for value in outlier_feature_indices.values():
    outlier_tuple_set.update(value)
X_copy_repair_indices = list(outlier_tuple_set)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

decision_tree = KNeighborsClassifier(n_neighbors=3)
decision_tree.fit(X_copy_inners, y_inners)

# 预测异常值
y_pred = decision_tree.predict(X_copy_repair)

# 替换异常值
y[X_copy_repair_indices] = y_pred
y_train = y[train_indices]
y_test = y[test_indices]

# subsection 重新在修复后的数据上训练decision_tree模型

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid_search.fit(X_train_copy, y_train)

# 获取最佳模型
decision_tree_repair = grid_search.best_estimator_
y_train_pred = decision_tree_repair.predict(X_train_copy)
y_test_pred = decision_tree_repair.predict(X_test_copy)

print("*" * 100)
# 训练样本中被decision_tree模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
print("加噪标签修复后，训练样本中被decision_tree模型错误分类的样本占总训练样本的比例：",
      len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被decision_tree模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
print("加噪标签修复后，测试样本中被decision_tree模型错误分类的样本占总测试样本的比例：",
      len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被decision_tree模型错误分类的样本
print("加噪标签修复后，完整数据集D中被decision_tree模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# # section 方案二：对X_copy中需要修复的元组进行特征修复（统计方法修复）
# #  需要修复的元组通过异常值检测器检测到的元组和decision_tree分类错误的元组共同确定（取并集）
#
# # subsection 按照特征中的异常值进行修复
#
# for key, value in outlier_feature_indices.items():
#     column_data = X_copy[:, key]
#     mean = np.mean(column_data)
#     X_copy[value, key] = mean
#
# X_train_copy = X_copy[train_indices]
# X_test_copy = X_copy[test_indices]
#
# # subsection 重新在修复后的数据上训练decision_tree模型
#
# # 使用网格搜索进行超参数优化
# grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
# grid_search.fit(X_train_copy, y_train)
#
# # 获取最佳模型
# decision_tree_repair = grid_search.best_estimator_
# y_train_pred = svm_repair.predict(X_train_copy)
# y_test_pred = svm_repair.predict(X_test_copy)
#
# print("*" * 100)
# # 训练样本中被decision_tree模型错误分类的样本
# wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
# print("加噪标签修复后，训练样本中被decision_tree模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
#
# # 测试样本中被decision_tree模型错误分类的样本
# wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
# print("加噪标签修复后，测试样本中被decision_tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
#
# # 整体数据集D中被decision_tree模型错误分类的样本
# print("加噪标签修复后，完整数据集D中被decision_tree模型错误分类的样本占总完整数据的比例：",
#       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))