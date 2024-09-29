import pandas as pd

df = pd.read_csv('covtype.csv')

# 保留前11列特征列和标签列（假设标签列是最后一列）
new_df = df.iloc[:, :11].join(df.iloc[:, -1])

# 将新数据集保存为新的CSV文件
new_df.to_csv('covtype_process.csv', index=False)