import pandas as pd

file_path = './apple_rule.csv'
data = pd.read_csv(file_path)
df = data[['GOAD', 'DeepSVDD', 'RCA', 'RePEN', 'SLAD', 'ICL', 'NeuTraL', 'DevNet'\
                                    ,'DeepSAD', 'RoSAS', 'PReNet', 'Quartile', 'Z_score', 'Success']].astype(bool)
true_ratio = df.mean()
print(true_ratio)

both_true_count = df[(df['ICL'] == True) & (df['Success'] == True)].shape[0]
total_count = df.shape[0]
both_true_ratio = both_true_count / total_count
print("属性ICL和属性Success都为True的元组比例：", both_true_ratio)