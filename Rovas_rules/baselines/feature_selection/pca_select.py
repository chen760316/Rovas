"""
方案一、二效果较好
选用不同的数据集展开实验
采用KNN修改异常值的标签
采用统计方法修复异常值的特征
使用SVM分类器进行实验
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# choice drybean数据集
file_path = "../../../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)
enc = LabelEncoder()
# 原始数据集D对应的Dataframe
data['Class'] = enc.fit_transform(data['Class'])
X = data.values[:, :-1]
y = data.values[:, -1]
categorical_features = [0, 6]

# choice obesity数据集
# file_path = "../../kaggle_datasets/Obesity_prediction/obesity_data.csv"
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

# choice wine-quality数据集 (svm模型下分类效果不好)
# file_path = "../../UCI_datasets/wine+quality/winequality-white.csv"
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
# categorical_features = []

# choice apple_quality数据集（使用SVM的默认模型，方差确定重要特征，修复特征中的异常最有效）
# file_path = "../../kaggle_datasets/Apple_Quality/apple_quality.csv"
# data = pd.read_csv(file_path)
# # 删除id列
# data = data.drop(data.columns[0], axis=1)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Quality'] = enc.fit_transform(data['Quality'])
# categorical_features = []
# X = data.values[:, :-1]
# y = data.values[:, -1]

# choice balita数据集 (使用SVM的默认模型，方差确定重要特征。特征数越少，修复特征中异常效果越好，可以考虑特征数较少的数据集)
# file_path = "../../kaggle_datasets/balita/data_balita.csv"
# data = pd.read_csv(file_path)
# # 从 data 中随机抽取 10% 的样本
# data = data.sample(frac=0.1, random_state=42)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Nutrition_Status'] = enc.fit_transform(data['Nutrition_Status'])
# data['Gender'] = enc.fit_transform(data['Gender'])
# categorical_features = [0, 1]
# X = data.values[:, :-1]
# y = data.values[:, -1]

# choice Iris数据集 (使用SVM的默认模型，方差确定重要特征。特征数越少，修复特征中异常效果越好，可以考虑特征数较少的数据集)
# file_path = "../../kaggle_datasets/Iris_Species/Iris.csv"
# data = pd.read_csv(file_path)
# # 删除id列
# data = data.drop(data.columns[0], axis=1)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Species'] = enc.fit_transform(data['Species'])
# categorical_features = []
# X = data.values[:, :-1]
# y = data.values[:, -1]

# choice Obesity数据集
# file_path = "../../kaggle_datasets/Obesity_prediction/obesity_data.csv"
# data = pd.read_csv(file_path)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['ObesityCategory'] = enc.fit_transform(data['ObesityCategory'])
# data['Gender'] = enc.fit_transform(data['Gender'])
# categorical_features = [0, 1, 5]
# X = data.values[:, :-1]
# y = data.values[:, -1]

# choice Wine dataset数据集 (效果不好)
# file_path = "../../kaggle_datasets/Wine dataset/Wine dataset.csv"
# data = pd.read_csv(file_path)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['class'] = enc.fit_transform(data['class'])
# categorical_features = [5, 13]
# X = data.values[:, 1:]
# y = data.values[:, 0]

# choice adult数据集 (SVM拟合大数据集速度很慢，可以对数据集截取后训练svm)
# file_path = "../../nosiy_datasets/adult/adult.csv"
# data = pd.read_csv(file_path)
# # 从 data 中随机抽取 20% 的样本
# data = data.sample(frac=0.2, random_state=42)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Income'] = enc.fit_transform(data['Income'])
# categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# feature_names = data.columns[:-1].tolist()
# for i in categorical_features:
#     column_name = feature_names[i]
#     data[column_name] = enc.fit_transform(data[column_name])
# X = data.values[:, :-1]
# y = data.values[:, -1]

# section 数据特征缩放
# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
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

# choice 借助PCA方法找到对分类器有重要影响的特征 (效果不好)
from sklearn.decomposition import PCA

svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_model = svm.SVC()
svm_model.fit(X_train_copy, y_train)
pca = PCA(n_components=0.95)  # 保留 95% 的方差
X_train_copy_copy = np.copy(X_train_copy)
X_pca = pca.fit_transform(X_train_copy_copy)
components = pca.components_
feature_importance = np.abs(components).sum(axis=0)  # 计算特征重要性
# 挑选最重要的 k 个特征
k = 6  # 选择前 10 个特征
top_k_indices = np.argsort(feature_importance)[-k:]  # 挑选最重要的 k 个特征对应的索引
print("最重要的 k 个特征的索引:", top_k_indices)

# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
# i = X_train_copy.shape[1]
# np.random.seed(1)
# categorical_names = {}
# svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
# # svm_model = svm.SVC()
# svm_model.fit(X_train_copy, y_train)
#
# for feature in categorical_features:
#     le = LabelEncoder()
#     le.fit(data_copy.iloc[:, feature])
#     data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
#     categorical_names[feature] = le.classes_
#
# explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
#                                                    categorical_features=categorical_features,
#                                                    categorical_names=categorical_names, kernel_width=3)
#
# predict_fn = lambda x: svm_model.predict_proba(x)
# exp = explainer.explain_instance(X_train[i], predict_fn, num_features=6)
# # 获取最具影响力的特征及其权重
# top_features = exp.as_list()
# important_features = []
# for feature_set in top_features:
#     feature_long = feature_set[0]
#     for feature in feature_names:
#         if set(feature).issubset(set(feature_long)):
#             important_features.append(feature)
#             break
#
# top_k_indices = [feature_names.index(feature_name) for feature_name in important_features]
# print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# choice 无模型(非参数)方法中的Permutation Feature Importance-slearn(效果未知)（速度慢）
# from sklearn.inspection import permutation_importance
# top_k_svm = 4
# svm_model = svm.SVC()
# svm_model.fit(X_train_copy, y_train)
# result = permutation_importance(svm_model, X_train_copy, y_train, n_repeats=10,random_state=42)
# permutation_importance = result.importances_mean
# top_k_indices = np.argpartition(-permutation_importance, top_k_svm)[:top_k_svm]
# print("Permutation Feature Importance-slearn检验的最有影响力的的前{}个属性的索引：{}".format(top_k_svm, top_k_indices))

# choice 使用SelectFromModel方法(效果不好)
# from sklearn.feature_selection import SelectFromModel
# svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_model.fit(X_train_copy, y_train)
# sfm = SelectFromModel(svm_model, threshold='mean', prefit=True)
# X_selected = sfm.transform(X_copy)
# # 获取选择的特征索引
# top_k_indices = sfm.get_support(indices=True)
# # 打印选择的特征索引
# print("SelectFromModel选择器选择的特征索引:", top_k_indices)

# choice 使用系数方法（效果不好）
# # 找到训练数据中有影响力的特征（最佳特证数是4，7，8，10）
# top_k_svm = 6
# svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_model.fit(X_train_copy, y_train)
# # 提取系数
# feature_importances_coef = np.abs(svm_model.coef_[0])
#
# # 对系数进行排序
# top_k_indices = np.argpartition(-feature_importances_coef, top_k_svm)[:top_k_svm]
# print("SVM模型选择的特征索引是：", top_k_indices)

# choice 借助方差判别有影响力的特征(效果还可以)
# if X_copy.shape[1] < 8:
#     top_k_var = X_copy.shape[1] // 2 + 1
# else:
#     top_k_var = X_copy.shape[1] // 2
# svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_model.fit(X_train_copy, y_train)
# variances = np.var(X_copy, axis=0)
# top_k_indices = np.argsort(variances)[-top_k_var:][::-1]
# print("方差最大的前{}个特征的索引：{}".format(top_k_var, top_k_indices))

# choice 使用RFE递归地训练模型并删除最不重要的特征 (效果不好)
# from sklearn.feature_selection import RFE
#
# # 创建模型
# svm_model = svm.SVC(kernel='linear', C=1.0)
# selector = RFE(svm_model, n_features_to_select=6)  # 选择前5个特征
# selector = selector.fit(X_train_copy, y_train)
# # 获取被选择的特征
# top_k_indices = np.where(selector.support_)[0]
# # 训练svm模型
# svm_model.fit(X_train_copy, y_train)
# print("选择的特征：", top_k_indices)

# section 找到loss(M, D, 𝑡) > 𝜆的元组

# choice 使用sklearn库中的hinge损失函数
# decision_values = svm_model.decision_function(X_copy)
# predicted_labels = np.argmax(decision_values, axis=1)
# # 计算每个样本的hinge损失
# num_samples = X_copy.shape[0]
# num_classes = svm_model.classes_.shape[0]
# hinge_losses = np.zeros((num_samples, num_classes))
# hinge_loss = np.zeros(num_samples)
# for i in range(num_samples):
#     correct_class = int(y[i])
#     for j in range(num_classes):
#         if j != correct_class:
#             loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
#             hinge_losses[i, j] = loss_j
#     hinge_loss[i] = np.max(hinge_losses[i])
#
# # 在所有加噪数据D中损失函数高于阈值的样本索引
# ugly_outlier_candidates = np.where(hinge_loss > 1)[0]
# # print("D中损失函数高于损失阈值的样本索引为：", ugly_outlier_candidates)

# choice 使用svm模型预测结果
y_p = svm_model.predict(X_copy)
ugly_outlier_candidates = np.where(y != y_p)[0]

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

# SECTION SVM模型的实现

# subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_clf = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)
train_label_pred = svm_clf.predict(X_train)
test_label_pred = svm_clf.predict(X_test)

# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
print("训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
print("测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
train_label_pred_noise = svm_model.predict(X_train_copy)
test_label_pred_noise = svm_model.predict(X_test_copy)

# 加噪训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
print("加噪训练样本中被SVM模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
print("加噪测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被SVM模型错误分类的样本
print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 方案一：对X_copy中需要修复的元组进行标签修复（knn方法）
#  需要修复的元组通过异常值检测器检测到的元组和SVM分类错误的元组共同确定（取并集）

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

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_copy_inners, y_inners)

# 预测异常值
y_pred = knn.predict(X_copy_repair)

# 替换异常值
y[X_copy_repair_indices] = y_pred
y_train = y[train_indices]
y_test = y[test_indices]

# subsection 重新在修复后的数据上训练SVM模型

svm_repair = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_repair = svm.SVC()
svm_repair.fit(X_train_copy, y_train)
y_train_pred = svm_repair.predict(X_train_copy)
y_test_pred = svm_repair.predict(X_test_copy)

print("*" * 100)
# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
print("加噪标签修复后，训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
print("加噪标签修复后，测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("加噪标签修复后，完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：",
      (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# # section 方案二：对X_copy中需要修复的元组进行特征修复（统计方法修复）
# #  需要修复的元组通过异常值检测器检测到的元组和SVM分类错误的元组共同确定（取并集）

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
# # subsection 重新在修复后的数据上训练SVM模型
#
# # svm_repair = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_repair = svm.SVC(kernel='linear', C=1.0, probability=True)
# svm_repair.fit(X_train_copy, y_train)
# y_train_pred = svm_repair.predict(X_train_copy)
# y_test_pred = svm_repair.predict(X_test_copy)
#
# print("*" * 100)
# # 训练样本中被SVM模型错误分类的样本
# wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
# print("加噪标签修复后，训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
#
# # 测试样本中被SVM模型错误分类的样本
# wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
# print("加噪标签修复后，测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
#
# # 整体数据集D中被SVM模型错误分类的样本
# print("加噪标签修复后，完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：",
#       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))