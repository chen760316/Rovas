import pandas as pd

from mlxtend.frequent_patterns import fpgrowth, association_rules

file_path = './obesity_rule.csv'
data = pd.read_csv(file_path)
frequent_item_sets = fpgrowth(data[['GOAD', 'DeepSVDD', 'RCA', 'RePEN', 'SLAD', 'ICL', 'NeuTraL', 'DevNet'\
                                   ,'DeepSAD', 'RoSAS', 'PReNet', 'Quartile', 'Z_score', 'Success']], min_support=0.001, use_colnames=True)
frequent_item_sets.sort_values(by='support', ascending=False, inplace=True)
frequent_item_sets.to_csv('fpgrowth_frequent.csv', sep='\t', index=False)
#print(frequent_item_sets)

rules = association_rules(frequent_item_sets, metric='lift', min_threshold=0.1)
rules = rules[(rules['consequent support'] >=0.5)]
rules.sort_values(by = 'lift',ascending=False,inplace=True)
# print(rules)
rules.to_csv('result/fpgrowth_rule.csv', sep='\t', index=False)

