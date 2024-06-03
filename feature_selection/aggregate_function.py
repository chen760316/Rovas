"""
三种聚合函数示例
"""
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 构造数据集
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
columns = [f'feature{i+1}' for i in range(20)]  # 创建特征列名称
data = {col: X[:, i] for i, col in enumerate(columns)}
data['label'] = y  # 添加标签列
df = pd.DataFrame(data)

"""outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃)实现，元组索引从0开始"""
threshold = 0.1
select_feature = 'feature3'
mean_feature_value = df[select_feature].mean()
std_feature_value = df[select_feature].std()
# 计算feature_3值与其他元组的差异
diff_feature = abs(df[select_feature].values.reshape(-1, 1) - df[select_feature].values)
diff_feature[np.diag_indices(len(df))] = threshold  # 将对角线元素（与自身比较）设为threshold
# 找到符合条件的元组
satisfying_indices = np.where((diff_feature >= threshold).all(axis=1))[0]
print("符合条件的元组索引为：", satisfying_indices)

"""imbalanced(𝐷, 𝑅, 𝑡.𝐴, 𝛿)实现，分组按照不同间隔区间划分"""
# 设置分组的间隔
interval = 0.01
# 初始化MinMaxScaler
scaler = MinMaxScaler()
# 对DataFrame的各个列进行归一化
normalized_df = df.copy()  # 复制一个新的DataFrame，以保留原始数据
normalized_df[df.columns] = scaler.fit_transform(df[df.columns])
# 对每列数据进行分组
bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
# 统计每列数据占据了多少个间隔
for column in normalized_df.columns:
    digitized = np.digitize(normalized_df[column], bins)
    unique_bins = np.unique(digitized)
    print(f"列 '{column}' 占据了 {len(unique_bins)} 个间隔")

"""SDomain(𝐷, 𝑅, 𝐴, 𝜎)的实现，判断属性的活动域是否很小"""
domain_df = df.copy()
# 四舍五入保存两位小数
rounded_df = domain_df.round(2)
# 统计每列的不同元素数量
unique_counts = rounded_df.nunique().sort_values()
print(unique_counts)