"""
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧ loss(M, D, 𝑡) > 𝜆 ∧ M𝑐 (𝑅, 𝐴,M) → ugly(𝑡)
Rovas对ugly outliers的检测能力
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
# file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"

data = pd.read_csv(file_path)

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 20000:
    data = data.sample(n=20000, random_state=42)

enc = LabelEncoder()
label_name = data.columns[-1]

# 原始数据集D对应的Dataframe
data[label_name] = enc.fit_transform(data[label_name])

# 检测非数值列
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# 为每个非数值列创建一个 LabelEncoder 实例
encoders = {}
for column in non_numeric_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

X = data.values[:, :-1]
y = data.values[:, -1]

# 统计不同值及其数量
unique_values, counts = np.unique(y, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"标签: {value}, 数量: {count}")

# 找到最小标签的数量
min_count = counts.min()
total_count = counts.sum()

# 计算比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")
min_count_index = np.argmin(counts)  # 找到最小数量的索引
min_label = unique_values[min_count_index]  # 对应的标签值

all_columns = data.columns.values.tolist()
feature_names = all_columns[:-1]
class_name = all_columns[-1]

# 找到分类特征的列名
categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
# 获取分类特征对应的索引
categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

# section 数据特征缩放以及添加噪声

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
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从加噪数据中生成加噪训练数据和加噪测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=all_columns)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

# section 找到有影响力的特征 M𝑐 (𝑅, 𝐴, M)
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
import re

i = len(feature_names)
np.random.seed(1)
categorical_names = {}
softmax_model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
softmax_model.fit(X_train_copy, y_train)

for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data_copy.iloc[:, feature])
    data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
    categorical_names[feature] = le.classes_

explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

predict_fn = lambda x: softmax_model.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names)//2)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
top_k_indices = [feature_names.index(name) for name in top_feature_names]
print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# section 找到loss(M, D, 𝑡) > 𝜆的元组

# choice 使用交叉熵损失函数
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

# 获取决策值
decision_values = softmax_model.decision_function(X_copy)
# 将决策值转换为适用于 Softmax 的二维数组
decision_values_reshaped = decision_values.reshape(-1, 1)  # 变成 (n_samples, 1)
# 应用 Softmax 函数（可以手动实现或使用 scipy）
y_pred = softmax(np.hstack((decision_values_reshaped, -decision_values_reshaped)), axis=1)
# 创建 OneHotEncoder 实例
encoder = OneHotEncoder(sparse=False)
# 预测y_test的值，并与y_train组合成为y_ground
y_test_pred = softmax_model.predict(X_test_copy)
y_ground = np.hstack((y_train, y_test_pred))
# 对y_ground进行独热编码
y_true = encoder.fit_transform(y_ground.reshape(-1, 1))
# 计算每个样本的损失
loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# 计算测试集平均多分类交叉熵损失
average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
bad_samples = np.where(loss_per_sample < average_loss)[0]
good_samples = np.where(loss_per_sample >= average_loss)[0]
ugly_outlier_candidates = bad_samples

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

# section 确定数据中的ugly outliers

outlier_tuple_set = set()
for value in outlier_feature_indices.values():
    outlier_tuple_set.update(value)
outlier_tuple_set.update(bad_samples)
X_copy_repair_indices = list(outlier_tuple_set)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# section 训练下游任务的softmax模型

# subsection 原始数据集上训练的softmax模型在训练集和测试集中分错的样本比例

print("*" * 100)
softmax_model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
softmax_model.fit(X_train, y_train)
train_label_pred = softmax_model.predict(X_train)

# 训练样本中被softmax模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != softmax_model.predict(X_train))[0]
print("训练样本中被softmax模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被softmax模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != softmax_model.predict(X_test))[0]
print("测试样本中被softmax模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被softmax模型错误分类的样本
print("完整数据集D中被softmax模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的softmax模型在训练集和测试集中分错的样本比例

print("*" * 100)
softmax_model_noise = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
softmax_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = softmax_model_noise.predict(X_train_copy)

# 加噪训练样本中被softmax模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != softmax_model_noise.predict(X_train_copy))[0]
print("加噪训练样本中被softmax模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被softmax模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != softmax_model_noise.predict(X_test_copy))[0]
print("加噪测试样本中被softmax模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被softmax模型错误分类的样本
print("完整数据集D中被softmax模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 全部加噪数据中被softmax分类器误分类的数量
label_pred = softmax_model_noise.predict(X_copy)
wrong_classify_indices = []
for i in range(len(X_copy)):
    if y[i] != label_pred[i]:
        wrong_classify_indices.append(i)
print("被误分类的样本数量：", len(wrong_classify_indices))

# section 检测ugly outliers的召回率
# ugly_found_by_detector = list(set(X_copy_repair_indices) & set(wrong_classify_indices))
ugly_found_by_detector = list(set(X_copy_repair_indices) & set(wrong_classify_indices))
print("召回的ugly outliers的数量：", len(ugly_found_by_detector))
print("ugly outliers的召回率为：", len(ugly_found_by_detector)/len(wrong_classify_indices))