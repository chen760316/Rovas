from scipy.io import arff
import pandas as pd

# 读取 ARFF 文件
data, meta = arff.loadarff('E:/Rovas/normal_experiment/datasets/real_outlier_varying_ratios/Wilt/Wilt_withoutdupl_05.arff')

# 将数据转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 打印数据
print(df.head())

df.to_csv('E:/Rovas/normal_experiment/datasets/real_outlier_varying_ratios/Wilt/Wilt_withoutdupl_05.csv', index=False)
