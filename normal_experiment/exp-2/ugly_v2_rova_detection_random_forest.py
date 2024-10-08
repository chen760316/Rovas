"""
𝑅(𝑡) ∧ M𝑜 (𝑡, D) ∧ 𝑋1 → ugly(𝑡)
Rovas在传统异常检测领域检测数据中的异常值的效果
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from deepod.models.tabular import PReNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# choice 选取数据集
# file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
file_path = "../datasets/real_outlier/optdigits.csv"
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

all_columns = data.columns.values.tolist()
feature_names = all_columns[:-1]
class_name = all_columns[-1]

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
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

# SECTION M𝑜 (𝑡, D) 针对元组异常的（弱）监督异常检测器PReNet

# subsection 设置训练和测试的弱监督样本
# 设置弱监督训练样本
# 找到所有标签为 1 的样本索引
semi_label_ratio = 0.1  # 设置已知的异常标签比例
positive_indices = np.where(y_train == min_label)[0]
# 随机选择 10% 的正样本
n_positive_to_keep = int(len(positive_indices) * semi_label_ratio)
selected_positive_indices = np.random.choice(positive_indices, n_positive_to_keep, replace=False)
# 创建用于异常检测器的训练标签
y_semi = np.zeros_like(y_train)  # 默认全为 0
y_semi[selected_positive_indices] = 1  # 设置选中的正样本为 1
# 创建用于异常检测器的测试标签
y_semi_test = np.zeros_like(y_test)
test_positive_indices = np.where(y_test == min_label)[0]
y_semi_test[test_positive_indices] = 1

# subsection 异常检测器训练
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

out_clf = PReNet(epochs=epochs, device=device, random_state=random_state)
out_clf.fit(X_train, y=y_semi)

out_clf_noise = PReNet(epochs=epochs, device=device, random_state=random_state)
out_clf_noise.fit(X_train_copy, y_semi)

# SECTION 借助异常检测器，在训练集上进行异常值检测

# subsection 从原始训练集中检测出异常值索引

print("*"*100)
train_scores = out_clf.decision_function(X_train)
train_pred_labels, train_confidence = out_clf.predict(X_train, return_confidence=True)
print("训练集中异常值判定阈值为：", out_clf.threshold_)
train_outliers_index = []
print("训练集样本数：", len(X_train))
for i in range(len(X_train)):
    if train_pred_labels[i] == 1:
        train_outliers_index.append(i)
train_correct_detect_samples = []
for i in range(len(X_train)):
    if train_pred_labels[i] == y_semi[i]:
        train_correct_detect_samples.append(i)
print("训练集中异常检测器的检测准确度：", len(train_correct_detect_samples)/len(X_train))
# 训练样本中的异常值索引
print("训练集中异常值索引：", train_outliers_index)
print("训练集中的异常值数量：", len(train_outliers_index))
print("训练集中的异常值比例：", len(train_outliers_index)/len(X_train))

# subsection 从原始测试集中检测出异常值索引

print("*"*100)
test_scores = out_clf.decision_function(X_test)
test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
print("测试集中异常值判定阈值为：", out_clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)
test_correct_detect_samples = []
for i in range(len(X_test)):
    if test_pred_labels[i] == y_semi_test[i]:
        test_correct_detect_samples.append(i)
print("测试集中异常检测器的检测准确度：", len(test_correct_detect_samples)/len(X_test))
# 训练样本中的异常值索引
print("测试集中异常值索引：", test_outliers_index)
print("测试集中的异常值数量：", len(test_outliers_index))
print("测试集中的异常值比例：", len(test_outliers_index)/len(X_test))

# section 从加噪数据集的训练集和测试集中检测出的异常值
# subsection 从加噪训练集中检测出异常值索引

print("*"*100)
train_scores_noise = out_clf_noise.decision_function(X_train_copy)
train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
train_outliers_index_noise = []
print("加噪训练集样本数：", len(X_train_copy))
for i in range(len(X_train_copy)):
    if train_pred_labels_noise[i] == 1:
        train_outliers_index_noise.append(i)
train_correct_detect_samples_noise = []
for i in range(len(X_train_copy)):
    if train_pred_labels_noise[i] == y_semi[i]:
        train_correct_detect_samples_noise.append(i)
print("训练集中异常检测器的检测准确度：", len(train_correct_detect_samples_noise)/len(X_train_copy))
# 训练样本中的异常值索引
print("加噪训练集中异常值索引：", train_outliers_index_noise)
print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))

# subsection 从加噪测试集中检测出异常值索引

print("*"*100)
test_scores_noise = out_clf_noise.decision_function(X_test_copy)
test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
test_outliers_index_noise = []
print("加噪测试集样本数：", len(X_test_copy))
for i in range(len(X_test_copy)):
    if test_pred_labels_noise[i] == 1:
        test_outliers_index_noise.append(i)
test_correct_detect_samples_noise = []
print(len(test_pred_labels_noise), len(y_test))
for i in range(len(X_test_copy)):
    if test_pred_labels_noise[i] == y_semi_test[i]:
        test_correct_detect_samples_noise.append(i)
print("测试集中异常检测器的检测准确度：", len(test_correct_detect_samples_noise)/len(X_test_copy))
# 训练样本中的异常值索引
print("加噪测试集中异常值索引：", test_outliers_index_noise)
print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))

# subsection 从整个加噪数据集中检测出异常值索引

print("*"*100)
scores_noise = out_clf_noise.decision_function(X_copy)
pred_labels_noise, confidence_noise = out_clf_noise.predict(X_copy, return_confidence=True)
outliers_index_noise = []
for i in range(len(X_copy)):
    if pred_labels_noise[i] == 1:
        outliers_index_noise.append(i)

# SECTION random forest模型的实现和准确度测试

# subsection 原始数据集上训练的random forest模型在训练集和测试集中分错的样本比例

print("*" * 100)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
train_label_pred = rf_model.predict(X_train)

# 训练样本中被random forest模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != rf_model.predict(X_train))[0]
print("训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被random forest模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != rf_model.predict(X_test))[0]
print("测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被random forest模型错误分类的样本
print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的random forest模型在训练集和测试集中分错的样本比例

print("*" * 100)
rf_model_noise = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight='balanced')
rf_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = rf_model_noise.predict(X_train_copy)

# 加噪训练样本中被random forest模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != rf_model_noise.predict(X_train_copy))[0]
print("加噪训练样本中被random forest模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被random forest模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != rf_model_noise.predict(X_test_copy))[0]
print("加噪测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被random forest模型错误分类的样本
print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section M𝑐 (𝑅, 𝐴,M) 确定有影响力的特征
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
import re

# 特征数取4或6
i = len(feature_names)
np.random.seed(1)
categorical_names = {}

for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data.iloc[:, feature])
    data.iloc[:, feature] = le.transform(data.iloc[:, feature])
    categorical_names[feature] = le.classes_
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)
# predict_proba 方法用于分类任务，predict 方法用于回归任务
predict_fn = lambda x: rf_model.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names)//2)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
top_k_indices = [feature_names.index(name) for name in top_feature_names]
print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# section 识别X_copy中需要修复的元组

# 异常检测器检测出的训练集和测试集中的异常值在原含噪数据D'中的索引
train_outliers_noise = train_indices[train_outliers_index_noise]
test_outliers_noise = test_indices[test_outliers_index_noise]
outliers_noise = np.union1d(train_outliers_noise, test_outliers_noise)

# 在加噪数据集D'上训练的random forest模型，其分类错误的样本在原含噪数据D'中的索引
train_wrong_clf_noise = train_indices[wrong_classified_train_indices_noise]
test_wrong_clf_noise = test_indices[wrong_classified_test_indices_noise]
wrong_clf_noise = np.union1d(train_wrong_clf_noise, test_wrong_clf_noise)

# outliers和分错样本的并集
train_union = np.union1d(train_outliers_noise, train_wrong_clf_noise)
test_union = np.union1d(test_outliers_noise, test_wrong_clf_noise)

# 加噪数据集D'上需要修复的值
# 需要修复的特征和标签值
X_copy_repair_indices = np.union1d(outliers_noise, wrong_clf_noise)

# section SDomain(𝐷, 𝑅, 𝐴, 𝜎) 选取活动域过小的特征

def calculate_made(data):
    median = np.median(data)  # 计算中位数
    abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
    mad = np.median(abs_deviation)  # 计算绝对误差均值
    made = 1.843 * mad
    return median, made

# 初始化MinMaxScaler
scaler = MinMaxScaler()
data_minmax = pd.read_csv(file_path)
data_minmax[data_minmax.columns] = scaler.fit_transform(data_minmax[data_minmax.columns])
# 设置分组的间隔
interval = 0.01
# 对每列数据进行分组
bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
columns_bins = {}
columns_bins_count = []
small_domain_features = []

for column in data_minmax.columns:
    digitized = np.digitize(data_minmax[column], bins)
    unique_bins, counts = np.unique(digitized, return_counts=True)
    columns_bins[column] = len(unique_bins)
    columns_bins_count.append(len(unique_bins))

for i in top_k_indices:
    select_feature = feature_names[i]
    selected_bins = columns_bins[select_feature]
    median, made = calculate_made(np.array(columns_bins_count))
    lower_threshold = median - 2 * made
    upper_threshold = median + 2 * made
    if selected_bins < lower_threshold:
        small_domain_features.append(i)
# 是有影响力的特征且活动域不小
filtered_important_feature_indices = [item for item in top_k_indices if item not in small_domain_features]

# section imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿) 确定特征A是否平衡

imbalanced_tuple_indices = set()

# 初始化MinMaxScaler
scaler_new = MinMaxScaler()
data_imbalance = pd.read_csv(file_path)
data_imbalance[data_imbalance.columns] = scaler_new.fit_transform(data_imbalance[data_imbalance.columns])

for feature in filtered_important_feature_indices:
    select_feature = feature_names[feature]
    # 对每列数据进行分组
    bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    digitized = np.digitize(data_imbalance[select_feature], bins)
    # 统计每个区间的计数
    unique_bins, counts = np.unique(digitized, return_counts=True)
    # 设置最小支持数差值
    median_imbalance, made_imbalance = calculate_made(counts)

    for t in X_copy_repair_indices:
        train_row_number = X_train.shape[0]
        ta = data_imbalance.iloc[t, feature]
        # 找到 ta 所在的间隔
        ta_bin = np.digitize([ta], bins)[0]
        # 找到 ta 所在间隔的计数
        ta_count = counts[unique_bins == ta_bin][0]
        lower_threshold = median_imbalance - 2 * made_imbalance
        upper_threshold = median_imbalance + 2 * made_imbalance
        if ta_count < lower_threshold or ta_count > upper_threshold:
            imbalanced_tuple_indices.add(t)

# section 计算交叉熵损失
# 获取概率值
y_pred = rf_model_noise.predict_proba(X_copy)[:, [1, 0]]
# 创建 OneHotEncoder 实例
encoder = OneHotEncoder(sparse=False)
# 拟合并转换 y_test
y_true = encoder.fit_transform(y.reshape(-1, 1))
# 计算每个样本的损失
loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# 计算测试集平均多分类交叉熵损失
average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
bad_samples = np.where(loss_per_sample > average_loss)[0]
good_samples = np.where(loss_per_sample <= average_loss)[0]

# 将结合了SDomain(𝐷, 𝑅, 𝐴, 𝜎)和imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)的元组，与异常检测器检测到的元组结合
outliers_detected_by_predicates = imbalanced_tuple_indices
outliers_detected_by_detector = set(outliers_index_noise)
outliers_detected_by_loss = bad_samples
outliers_detected_by_loss_list = outliers_detected_by_loss.tolist()
X_copy_repair_indices = list(outliers_detected_by_predicates.intersection(outliers_detected_by_detector))
X_copy_repair_indices = list(set(X_copy_repair_indices).intersection(set(outliers_detected_by_loss_list)))
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# section 汇总加噪样本中检测到的outliers

y_pred = np.zeros_like(y)
y_pred[X_copy_repair_indices] = 1
y_train_pred = y_pred[train_indices]
y_test_pred = y_pred[test_indices]
# 统计不同值及其数量
unique_values, counts = np.unique(y_test_pred, return_counts=True)
# 输出结果
for value, count in zip(unique_values, counts):
    print(f"预测的测试集标签: {value}, 预测的标签数量: {count}")
# 找到预测的最小标签的数量
min_count = counts.min()
total_count = counts.sum()
# 计算预测的最少标签的比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")

# section 使用各种评价指标评价Rovas检测到的outliers

"""Accuracy指标"""
print("*" * 100)
print("Rovas在加噪测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

"""Precision/Recall/F1指标"""
print("*" * 100)

# average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
# average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
# average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
# average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

print("Rovas在加噪测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
print("Rovas在加噪测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
print("Rovas在加噪测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

"""ROC-AUC指标"""
print("*" * 100)
roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')  # 一对多方式
print("Rovas在加噪测试集中的ROC-AUC分数：" + str(roc_auc_test))

"""PR AUC指标"""
print("*" * 100)
# 计算预测概率
y_scores = 1 / (1 + np.exp(-test_scores))
# 计算 Precision 和 Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
# 计算 PR AUC
pr_auc = auc(recall, precision)
print("半监督异常检测器在原始测试集中的PR AUC 分数:", pr_auc)

"""AP指标"""
print("*" * 100)
# 计算预测概率
y_scores = 1 / (1 + np.exp(-test_scores))
# 计算 Average Precision
ap_score = average_precision_score(y_test, y_scores)
print("无监督异常检测器在原始测试集中的AP分数:", ap_score)