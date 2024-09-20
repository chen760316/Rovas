from deepod.models.tabular import RCA
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from scipy import stats
from scipy.stats import multivariate_normal

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# SECTION 数据预处理
# file_path = "../kaggle_datasets/Obesity_prediction/obesity_data.csv"
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
# X = StandardScaler().fit_transform(X)
# 记录原始索引

# file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]

file_path = "../kaggle_datasets/Apple_Quality/apple_quality.csv"
data = pd.read_csv(file_path)
# 删除id列
data = data.drop(data.columns[0], axis=1)
enc = LabelEncoder()
# 原始数据集D对应的Dataframe
data['Quality'] = enc.fit_transform(data['Quality'])
categorical_features = []
X = data.values[:, :-1]
y = data.values[:, -1]
X = StandardScaler().fit_transform(X)
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=42)

# SECTION 训练异常检测器
clf = RCA(epochs=epochs, device=device, act='LeakyReLU')
clf.fit(X_train)

# SECTION 检测测试集中的异常值
test_scores = clf.decision_function(X_test)
test_pred_labels, test_confidence = clf.predict(X_test, return_confidence=True)
print("测试集中异常值判定阈值为：", clf.threshold_)
test_outliers_index = []
print("测试集样本数：", len(X_test))
for i in range(len(X_test)):
    if test_pred_labels[i] == 1:
        test_outliers_index.append(i)
# 训练样本中的异常值索引
print("测试集中异常值索引：", test_outliers_index)
print("测试集中的异常值数量：", len(test_outliers_index))

# SECTION 训练SVM分类器
# svm_model = svm.SVC()
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)

svm_model.fit(X_train, y_train)
test_label_pred = svm_model.predict(X_test)
wrong_classified_indices = np.where(y_test != svm_model.predict(X_test))[0]
print("测试集中分类错误的样本数量：", len(wrong_classified_indices))

# SECTION 使用多分类交叉熵损失函数
# decision_values = svm_model.decision_function(X_test)
# # 应用 Softmax 函数
# y_pred = softmax(decision_values, axis=1)
# # 创建 OneHotEncoder 实例
# encoder = OneHotEncoder(sparse=False)
# # 拟合并转换 y_test
# y_true = encoder.fit_transform(y_test.reshape(-1, 1))
# # 计算每个样本的损失
# loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# # 计算测试集平均多分类交叉熵损失
# average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
# bad_samples = np.where(loss_per_sample > average_loss)[0]
# # 测试样本中的bad outliers索引
# bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
# print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))

# section 使用二元交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    # 防止对数计算中的零
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    losses = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    loss = np.mean(losses)
    return losses, loss

def sigmoid(decision_values):
    return 1 / (1 + np.exp(-decision_values))

decision_values = svm_model.decision_function(X_test)
y_prediction = svm_model.predict(X_test)
# 计算概率
probabilities = sigmoid(decision_values)
print("属于类 1 的概率:", probabilities)
print("属于类 0 的概率:", 1 - probabilities)  # 属于类 0 的概率

loss_per_sample, average_loss = binary_cross_entropy(y_test, probabilities)
bad_samples = np.where(loss_per_sample > average_loss)[0]
# 测试样本中的bad outliers索引
bad_outliers_index = np.intersect1d(test_outliers_index, bad_samples)
print("检测出的outliers中bad outliers的数量：", len(bad_outliers_index))

# section 测试集中离群值中被SVM分类器误分类的数量
wrong_classify_indices = []
for i in test_outliers_index:
    true_label = y_test[i]
    if true_label != test_label_pred[i]:
        wrong_classify_indices.append(i)
print("检测出的outliers中被SVM分类器误分类的数量(ugly outliers的数量)：", len(wrong_classify_indices))

# section 判断ugly outliers和正常值在有影响力的特征中的差异
# ugly outliers
# top_k_indices = [3, 7, 1, 13, 0, 12]
top_k_indices = [0, 1, 2, 3, 4, 5]
feature_indices = np.array(top_k_indices)
ugly_outlier_samples = X_test[wrong_classify_indices, :]
ugly_outlier_samples = ugly_outlier_samples[:, top_k_indices]

# normal samples
# 首先，创建一个布尔索引数组，初始时所有元素都设置为True
mask = np.ones(len(X_test), dtype=bool)
# 然后将 outlier 索引对应的位置设置为False
mask[bad_outliers_index] = False
# 使用布尔索引数组来选择非outlier的样本
normal_samples = X_test[mask]
normal_samples = normal_samples[:, top_k_indices]

# bad outliers
bad_outlier_samples = X_test[bad_outliers_index, :]
bad_outlier_samples = bad_outlier_samples[:, top_k_indices]

# ks_stat, p_values = stats.ks_2samp(ugly_outlier_samples, normal_samples)
# print("KS-statistical:", ks_stat)
# print("p-value:", p_values)

# 取特征列对应的样本
# ugly_BMI = ugly_outlier_samples[:, -2]
# normal_BMI = normal_samples[:, -2]
# ks_stat_BMI, p_values_BMI = stats.ks_2samp(ugly_BMI, normal_BMI)
# print(ugly_BMI)
# print(normal_BMI)
# print("BMI特征的KS-statistical:", ks_stat_BMI)
# print("BMI特征的p-value:", p_values_BMI)
#
# 取特征列对应的样本
# ugly_weight = ugly_outlier_samples[:, -3]
# normal_weight = normal_samples[:, -3]
# ks_stat_weight, p_values_weight = stats.ks_2samp(ugly_weight, normal_weight)
# print("Weight特征的KS-statistical:", ks_stat_weight)
# print("Weight特征的p-value:", p_values_weight)
# print(ugly_weight)
# print(normal_weight)

# modified_z_score方法
# def mad_based_outlier(points, thresh=3.5):
#     if len(points.shape) == 1:
#         points = points[:,None]
#     median = np.median(points, axis=0)
#     diff = np.sum((points - median)**2, axis=-1)
#     diff = np.sqrt(diff)
#     med_abs_deviation = np.median(diff)
#     modified_z_score = 0.6745 * diff / med_abs_deviation
#     return modified_z_score > thresh
#
# print(mad_based_outlier(normal_BMI))
# print(mad_based_outlier(ugly_BMI))

# 标准偏差法
# mean = normal_BMI.mean()
# std = normal_BMI.std()
#
# upper_bound = mean + 3 * std
# lower_bound = mean - 3 * std
# print("上边界为：", upper_bound)
# print("下边界为：", lower_bound)

# section 拟合正常点的分布，判断异常值是否落在该分布内

# 计算数据的均值和协方差矩阵
mean = np.mean(normal_samples, axis=0)
cov = np.cov(normal_samples, rowvar=False)

# 创建多元正态分布对象
distribution = multivariate_normal(mean, cov)

# 计算异常点在拟合的分布下的概率密度
probability_density = distribution.pdf(ugly_outlier_samples)

# 打印概率密度
print("异常点的概率密度:", probability_density)

# 设定概率密度阈值
threshold = 0.01

# 判断异常点是否在正常分布内
for i, density in enumerate(probability_density):
    # print(f"样本 {i} 的概率密度: {density}")
    if density < threshold:
        print(f'异常点{i}不在正常分布内')
    else:
        print(f'异常点{i}在正常分布内')