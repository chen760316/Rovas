"""
没有外部数据源可用，或者外部数据源中没有匹配项时如何替代异常值
异常值检测与处理是数据预处理中的重要环节。常见方法包括统计方法（如Z-score、IQR）、
机器学习方法（如支持向量机）、以及深度学习方法（如自编码器）。
统计方法利用数据的统计特性（如均值、方差、中位数）进行异常值检测，适用于简单数据集。
一、统计方法
 1.1替换为均值、中位数或众数
    均值：对于数值型数据，可以用列的均值替换异常值。
    中位数：中位数是对异常值不敏感的替代值，适用于数值型数据。
    众数：对于分类数据，可以用最频繁出现的值（众数）替换异常值
 1.2 插值法
    时间序列数据或连续数据，可以使用插值方法填充缺失值或异常值
 1.3 借助数据分布
    对数据进行分布分析，使用数据的分布特征来检测异常值，借助均值或中位数进行修复
二、机器学习方法
 数据集较大且有多个特征，可以训练一个模型来预测缺失或异常的值
"""

import pandas as pd
import numpy as np

# section 1.1 使用均值，中位数或众数替换缺失值

# 假设异常值标记为NaN
def replace_with_statistics(data_df, column_name):
    # 计算均值，中位数或众数
    mean_value = data_df[column_name].mean()
    median_value = data_df[column_name].median()
    mode_value = data_df[column_name].mode()[0]

    # 使用均值，中位数和众数替换异常值
    data_df[column_name].replace("NaN", mean_value, inplace=True)
    # data_df[column_name].replace(-9999, median_value, inplace=True)
    # data_df[column_name].replace(-9999, mode_value, inplace=True)

    return data_df

# 读取数据
file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data_df = pd.read_excel(file_path)

# 替换异常值
data_df = replace_with_statistics(data_df, 'your_column_name')

# 保存更新后的dataframe
# data_df.to_excel("../UCI_datasets/dry+bean+dataset/DryBeanDataset/Updated_Dry_Bean_Dataset.xlsx", index=False)

# section 1.2 使用线性插值替换缺失值

def replace_with_interpolation(data_df, column_name):
    # 执行线性插值
    data_df[column_name] = data_df[column_name].replace("NaN", pd.NA)
    data_df[column_name] = data_df[column_name].interpolate()
    return data_df

# 使用插值修复异常值
data_df = replace_with_interpolation(data_df, 'your_column_name')

# section 1.3 使用3sigma原则的数据分布判别异常值，借助均值进行修复

def replace_with_distribution(data_df, column_name):
    # Define range within standard deviation
    mean = data_df[column_name].mean()
    std_dev = data_df[column_name].std()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    # Replace outliers
    data_df[column_name] = data_df[column_name].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)

    # Optionally replace NaNs with mean/median
    data_df[column_name].fillna(mean, inplace=True)

    return data_df

# section 2 使用线性回归替换异常值

# 训练模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
X = data_df[['feature1', 'feature2']]  # Features
y = data_df['target']  # Target

# Remove rows with missing target
X = X[~y.isna()]
y = y.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
data_df.loc[data_df['target'].isna(), 'target'] = model.predict(X_test)

# 用模型预测
def predict_and_replace(data_df, feature_columns, target_column, model):
    # Prepare data for prediction
    X_missing = data_df[data_df[target_column].isna()][feature_columns]

    # Predict and replace
    data_df.loc[data_df[target_column].isna(), target_column] = model.predict(X_missing)

    return data_df

# Predict and replace missing values
data_df = predict_and_replace(data_df, ['feature1', 'feature2'], 'target', model)
