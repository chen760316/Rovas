"""
方案一效果较好，方案二效果不好
采用KNN修改异常值的标签
采用统计方法修复异常值的特征
使用SVM分类器进行实验
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
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 在特征中注入高斯随机噪声
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
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从含噪数据中生成训练数据和测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)
# print("训练集中的噪声样本为：", train_noise)
# print("测试集中的噪声样本为：", test_noise)

# SECTION M𝑜 (𝑡, D) 针对元组异常的无监督异常检测器GOAD
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
clf_gold = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold.fit(X_train, y=None)

# SECTION 借助异常检测器，在训练集上进行异常值检测。
#  经过检验，加入高斯噪声会影响异常值判别

# subsection 从原始数据集的训练集和测试集中检测出的异常值

out_clf = clf_gold

# 从原始训练集中检测出异常值索引

train_scores = out_clf.decision_function(X_train)
train_pred_labels, train_confidence = out_clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", out_clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))
print("训练集中的异常值比例：", len(train_outliers_index)/len(X_train))

# 从原始测试集中检测出异常值索引

test_scores = out_clf.decision_function(X_test)
test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
print("测试集中异常值判定阈值为：", out_clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)
# 训练样本中的异常值索引
print("测试集中异常值索引：", test_outliers_index)
print("测试集中的异常值数量：", len(test_outliers_index))
print("测试集中的异常值比例：", len(test_outliers_index)/len(X_test))

# subsection 从加噪数据集的训练集和测试集中检测出的异常值

clf_gold_noise = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_gold_noise.fit(X_train_copy, y=None)
out_clf_noise = clf_gold_noise

# 从加噪训练集中检测出异常值索引

train_scores_noise = out_clf_noise.decision_function(X_train_copy)
train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
train_outliers_index_noise = []
print("加噪训练集样本数：", len(X_train_copy))
for i in range(len(X_train_copy)):
    if train_pred_labels_noise[i] == 1:
        train_outliers_index_noise.append(i)
# 训练样本中的异常值索引
print("加噪训练集中异常值索引：", train_outliers_index_noise)
print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))

# 从加噪测试集中检测出异常值索引

test_scores_noise = out_clf_noise.decision_function(X_test_copy)
test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
test_outliers_index_noise = []
print("加噪测试集样本数：", len(X_test_copy))
for i in range(len(X_test_copy)):
    if test_pred_labels_noise[i] == 1:
        test_outliers_index_noise.append(i)
# 训练样本中的异常值索引
print("加噪测试集中异常值索引：", test_outliers_index_noise)
print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))


# SECTION 随机森林模型的实现

# subsection 原始数据集上训练的随机森林模型在训练集和测试集中分错的样本比例

print("*" * 100)

# # 设置随机森林参数
# params = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
#
# # 使用网格搜索进行超参数优化
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
#
# # 获取最佳参数
# best_params = grid_search.best_params_
#
# # 使用最佳参数定义新的随机森林模型
# rf_clf = RandomForestClassifier(random_state=42, **best_params)
#
# # 获取最佳模型
# rf_classifier = grid_search.best_estimator_

rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)
rf_classifier.fit(X_train, y_train)
train_label_pred = rf_classifier.predict(X_train)
test_label_pred = rf_classifier.predict(X_test)

# 训练样本中被随机森林模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
print("训练样本中被随机森林模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被随机森林模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
print("测试样本中被随机森林模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被随机森林模型错误分类的样本
print("完整数据集D中被随机森林模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的随机森林模型在训练集和测试集中分错的样本比例

print("*" * 100)

# # 使用网格搜索进行超参数优化
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
#
# # 获取最佳模型
# rf_classifier_noise = grid_search.best_estimator_

rf_classifier_noise = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)
rf_classifier_noise.fit(X_train_copy, y_train)
train_label_pred_noise = rf_classifier_noise.predict(X_train_copy)
test_label_pred_noise = rf_classifier_noise.predict(X_test_copy)

# 加噪训练样本中被随机森林模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
print("加噪训练样本中被随机森林模型错误分类的样本占总加噪训练样本的比例：",
      len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被随机森林模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
print("加噪测试样本中被随机森林模型错误分类的样本占总测试样本的比例：",
      len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被随机森林模型错误分类的样本
print("完整数据集D中被随机森林模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 识别X_copy中需要修复的元组

# 异常检测器检测出的训练集和测试集中的异常值在原含噪数据D'中的索引
train_outliers_noise = train_indices[train_outliers_index_noise]
test_outliers_noise = test_indices[test_outliers_index_noise]
outliers_noise = np.union1d(train_outliers_noise, test_outliers_noise)

# 在加噪数据集D'上训练的随机森林模型，其分类错误的样本在原含噪数据D'中的索引
train_wrong_clf_noise = train_indices[wrong_classified_train_indices_noise]
test_wrong_clf_noise = test_indices[wrong_classified_test_indices_noise]
wrong_clf_noise = np.union1d(train_wrong_clf_noise, test_wrong_clf_noise)

# outliers和分错样本的并集
train_union = np.union1d(train_outliers_noise, train_wrong_clf_noise)
test_union = np.union1d(test_outliers_noise, test_wrong_clf_noise)

# 加噪数据集D'上需要修复的值
# 需要修复的特征和标签值
X_copy_repair_indices = np.union1d(outliers_noise, wrong_clf_noise)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# # section 方案一：对X_copy中需要修复的元组进行标签修复（knn方法）
# #  需要修复的元组通过异常值检测器检测到的元组和SVM分类错误的元组共同确定（取并集）
#
# # subsection 尝试修复异常数据的标签
#
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_copy_inners, y_inners)
#
# # 预测异常值
# y_pred = knn.predict(X_copy_repair)
#
# # 替换异常值
# y[X_copy_repair_indices] = y_pred
# y_train = y[train_indices]
# y_test = y[test_indices]
#
# # subsection 重新在修复后的数据上训练随机森林模型
#
# rf_classifier_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)
# rf_classifier_repair.fit(X_train_copy, y_train)
# # svm_repair = svm.SVC(kernel='linear', C=1.0, probability=True)
# # svm_repair.fit(X_train_copy, y_train)
# y_train_pred = rf_classifier_repair.predict(X_train_copy)
# y_test_pred = rf_classifier_repair.predict(X_test_copy)
#
# print("*" * 100)
# # 训练样本中被随机森林模型错误分类的样本
# wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
# print("加噪标签修复后，训练样本中被随机森林模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
#
# # 测试样本中被随机森林模型错误分类的样本
# wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
# print("加噪标签修复后，测试样本中被随机森林模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
#
# # 整体数据集D中被随机森林模型错误分类的样本
# print("加噪标签修复后，完整数据集D中被随机森林模型错误分类的样本占总完整数据的比例：",
#       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# section 方案二：对X_copy中需要修复的元组进行特征修复（统计方法修复）
#  需要修复的元组通过异常值检测器检测到的元组和SVM分类错误的元组共同确定（取并集）

rf_classifier_repair = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)
rf_classifier_repair.fit(X_train_copy, y_train)

# subsection 确定有影响力的特征
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)

# 特征数取4或6
i = 16
np.random.seed(1)
categorical_features = [0, 6]
categorical_names = {}
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=feature_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
# predict_proba 方法用于分类任务，predict 方法用于回归任务
predict_fn = lambda x: rf_classifier.predict_proba(x)
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

# subsection 确定异常值，使用2MADe方法确定有影响力的列中的异常值，将异常值的部分设置为np.nan
for i in range(X_copy.shape[1]):
    if i in top_k_indices:
        column_data = X_copy[:, i]
        mean = np.mean(column_data)
        # 将所有需要修复的行对应的列位置的元素替换为均值
        intersection = X_copy_repair_indices
        X_copy[intersection, i] = mean

# subsection 使用knn修复所有被标记为nan的异常特征

X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]

# subsection 重新在修复后的数据上训练随机森林模型

y_train_pred = rf_classifier_repair.predict(X_train_copy)
y_test_pred = rf_classifier_repair.predict(X_test_copy)

print("*" * 100)
# 训练样本中被随机森林模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
print("加噪标签修复后，训练样本中被随机森林模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被随机森林模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
print("加噪标签修复后，测试样本中被随机森林模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被随机森林模型错误分类的样本
print("加噪标签修复后，完整数据集D中被随机森林模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))