from scipy.io import arff
import pandas as pd

# 读取 ARFF 文件
data, meta = arff.loadarff('E:/Rovas/Rovas_rules/baselines/datasets/odds/seismic/seismic-bumps.arff')

# 将数据转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 打印数据
print(df.head())

df.to_csv('E:/Rovas/Rovas_rules/baselines/multi_class_datasets/open_source_data/seismic.csv', index=False)
