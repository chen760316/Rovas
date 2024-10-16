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

# file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
# data = pd.read_excel(file_path)
# enc = LabelEncoder()
# # 原始数据集D对应的Dataframe
# data['Class'] = enc.fit_transform(data['Class'])
# X = data.values[:, :-1]
# y = data.values[:, -1]

# file_path = "../Rovas_rules/baselines/multi_class_datasets/open_source_data/credit_card_fraud.csv"
# data = pd.read_csv(file_path)

file_path = "../normal_experiment/datasets/multi_class/student/Student.csv"
data = pd.read_csv(file_path)

label_name = data.columns[-1]
# 统计不同标签的数量
label_counts = data[label_name].value_counts()

print("不同标签及其数量:")
print(label_counts)

# 找到数据量最少的标签及其数量
min_label = label_counts.idxmin()  # 最少标签的名称
min_count = label_counts.min()      # 最少标签的数量

# 计算所有标签的总数量
total_count = label_counts.sum()

# 计算最少标签的比例
min_ratio = min_count / total_count if total_count > 0 else 0

print(f"数据量最少的标签: {min_label}, 数量: {min_count}")
print(f"该标签占所有标签的比例: {min_ratio:.2%}")