"""
探究SVM在异常检测二分类数据集上的检测性能(测试集)
超过两万行的数据，为了快速训练SVM均随机采样到两万行
"""

from collections import Counter
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
from deepod.models.tabular import DeepSVDD
from deepod.models.tabular import RCA
from deepod.models import REPEN, SLAD, ICL, NeuTraL
from deepod.models.tabular import DevNet
from deepod.models import DeepSAD, RoSAS, PReNet

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理，输入异常检测数据集

# subsection multi_class_to_outlier
# file_path = "datasets/multi_class_to_outlier/drybean_outlier.csv"  # 原始样本分类准确度1，加噪样本分类准确度1
# file_path = "datasets/multi_class_to_outlier/adult_outlier.csv"  # 原始样本分类准确度0.764，加噪样本分类准确度0.764
file_path = "datasets/multi_class_to_outlier/apple_outlier.csv"  # 原始样本分类准确度0.753，加噪样本分类准确度0.738
# file_path = "datasets/multi_class_to_outlier/balita_outlier.csv"  # 原始样本分类准确度0.889，加噪样本分类准确度0.889
# file_path = "datasets/multi_class_to_outlier/Iris_outlier.csv"  # 原始样本分类准确度1，加噪样本分类准确度0.978
# file_path = "datasets/multi_class_to_outlier/obesity_outlier.csv"  # 原始样本分类准确度0.993，加噪样本分类准确度0.977
# file_path = "datasets/multi_class_to_outlier/wine_outlier.csv"  # 原始样本分类准确度1，加噪样本分类准确度1

# subsection multi_class
# file_path = "datasets/multi_class/drybean.xlsx"  # 原始样本分类准确度0.926，加噪样本分类准确度0.878，用pd.read_excel读取
# file_path = "datasets/multi_class/adult.csv"  # 原始样本分类准确度0.764，加噪样本分类准确度0.764
# file_path = "datasets/multi_class/apple.csv"  # 原始样本分类准确度0.753，加噪样本分类准确度0.738
# file_path = "datasets/multi_class/balita.csv"  # 原始样本分类准确度0.792，加噪样本分类准确度0.614
# file_path = "datasets/multi_class/Iris.csv"  # 原始样本分类准确度1，加噪样本分类准确度0.956
# file_path = "datasets/multi_class/obesity.csv"  # 原始样本分类准确度0.980，加噪样本分类准确度0.890
# file_path = "datasets/multi_class/wine.csv"  # 原始样本分类准确度0.510，加噪样本分类准确度0.498

# subsection real_outlier
# file_path = "datasets/real_outlier/annthyroid.csv"  # 原始样本分类准确度0.946，加噪样本分类准确度0.932
# file_path = "datasets/real_outlier/breastw.csv"  # 原始样本分类准确度0.961，加噪样本分类准确度0.961
# file_path = "datasets/real_outlier/Cardiotocography.csv"  # 原始样本分类准确度0.923，加噪样本分类准确度0.874
# file_path = "datasets/real_outlier/credit_card_fraud.csv"  # 原始样本分类准确度0.999，加噪样本分类准确度0.999
# file_path = "datasets/real_outlier/optdigits.csv"  # 原始样本分类准确度0.999，加噪样本分类准确度0.991
# file_path = "datasets/real_outlier/PageBlocks.csv"  # 原始样本分类准确度0.946，加噪样本分类准确度0.917
# file_path = "datasets/real_outlier/pendigits.csv"  # 原始样本分类准确度0.996，加噪样本分类准确度0.994
# file_path = "datasets/real_outlier/satellite.csv"  # 原始样本分类准确度0.876，加噪样本分类准确度0.856
# file_path = "datasets/real_outlier/shuttle.csv"  # 原始样本分类准确度0.997，加噪样本分类准确度0.997
# file_path = "datasets/real_outlier/Waveform.csv"  # 原始样本分类准确度0.970，加噪样本分类准确度0.970
# file_path = "datasets/real_outlier/WPBC.csv"  # 原始样本分类准确度0.953，加噪样本分类准确度0.942

# subsection real_outlier_multi_class
# file_path = "datasets/real_outlier_multi_class/breastw.csv"  # 原始样本分类准确度0.805，加噪样本分类准确度0.795
# file_path = "datasets/real_outlier_multi_class/optdigits.csv"  # 原始样本分类准确度0.975，加噪样本分类准确度0.938
# file_path = "datasets/real_outlier_multi_class/PageBlocks.csv"  # 原始样本分类准确度0.968，加噪样本分类准确度0.948
# file_path = "datasets/real_outlier_multi_class/pendigits.csv"  # 原始样本分类准确度0.981，加噪样本分类准确度0.899
# file_path = "datasets/real_outlier_multi_class/satellite.csv"  # 原始样本分类准确度0.864，加噪样本分类准确度0.823
# file_path = "datasets/real_outlier_multi_class/shuttle.csv"  # 原始样本分类准确度0.977，加噪样本分类准确度0.920
# file_path = "datasets/real_outlier_multi_class/waveform.csv"  # 原始样本分类准确度0.875，加噪样本分类准确度0.847
# file_path = "datasets/real_outlier_multi_class/yeast.csv"  # 原始样本分类准确度0.575，加噪样本分类准确度0.544

# subsection synthetic_outlier
# file_path = "datasets/synthetic_outlier/apple_global_0.2.csv"  # 原始样本分类准确度0.803，加噪样本分类准确度0.803
# file_path = "datasets/synthetic_outlier/iris_cluster_0.15.csv"  # 原始样本分类准确度1，加噪样本分类准确度1
# file_path = "datasets/synthetic_outlier/optdigits_local_0.3.csv"  # 原始样本分类准确度0.692，加噪样本分类准确度0.692
# file_path = "datasets/synthetic_outlier/Waveform_cluster_0.05.csv"  # 原始样本分类准确度0.995，加噪样本分类准确度0.982
# file_path = "datasets/synthetic_outlier/wine_local_0.4.csv"  # 原始样本分类准确度0.608，加噪样本分类准确度0.608


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

# X = data.values[:, 1:]
# y = data.values[:, 0]

# 统计不同值及其数量
unique_values, counts = np.unique(y, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"标签: {value}, 数量: {count}")
categorical_features = [0, 6]

# 找到最小标签的数量
min_count = counts.min()
total_count = counts.sum()

# 计算比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")
min_count_index = np.argmin(counts)  # 找到最小数量的索引
min_label = unique_values[min_count_index]  # 对应的标签值

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

# SECTION SVM模型的实现

# subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
train_label_pred = svm_model.predict(X_train)
test_label_pred = svm_model.predict(X_test)

# 训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
print("训练样本中分类准确度：", 1-len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
print("测试样本中分类准确度：", 1-len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被SVM模型错误分类的样本
print("完整数据集D中分类准确度：",
      1-(len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model_noise = svm.SVC(kernel='linear', C=1.0, probability=True)
svm_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = svm_model_noise.predict(X_train_copy)
test_label_pred_noise = svm_model_noise.predict(X_test_copy)

# 加噪训练样本中被SVM模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
print("加噪训练样本中分类准确度：", 1-len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被SVM模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
print("加噪测试样本中分类准确度：", 1-len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被SVM模型错误分类的样本
print("完整数据集D中分类准确度：",
      1-(len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))