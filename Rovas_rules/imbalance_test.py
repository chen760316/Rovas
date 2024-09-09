import pandas as pd

# 读取CSV文件到DataFrame
df = pd.read_csv('../kaggle_datasets/balita/data_balita.csv')

# 统计标签列中不同标签的出现次数
label_counts = df['Gender'].value_counts()

# 打印统计结果
print(label_counts)

file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)

# 统计标签列中不同标签的出现次数
class_counts = data['Class'].value_counts()

# 打印统计结果
print(class_counts)