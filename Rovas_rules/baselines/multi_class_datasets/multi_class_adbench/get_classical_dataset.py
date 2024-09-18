from sklearn.datasets import fetch_openml
import pandas as pd

# 从 OpenML 加载 annthyroid 数据集
data = fetch_openml(name='Annthyroid', version=1)

# 将数据集转换为 DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['class'] = data.target

# 打印数据集的信息
print(df.head())
print(df['class'].value_counts())  # 查看类别分布