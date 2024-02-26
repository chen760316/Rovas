import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

file_path = './balita_rule.csv'
data = pd.read_csv(file_path)
support_list = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
for support in support_list:
    freq_file_path = 'result/apriority_frequent_%.3f.csv'%support
    rule_file_path = 'result/apriority_rule_%.3f.csv'%support
    frequent_item_sets = apriori(data[['GOAD', 'DeepSVDD', 'RCA', 'RePEN', 'SLAD', 'ICL', 'NeuTraL', 'DevNet'\
                                    ,'DeepSAD', 'RoSAS', 'PReNet', 'Quartile', 'Z_score', 'Success']], min_support=support, use_colnames=True)
    frequent_item_sets.sort_values(by='support', ascending=False, inplace=True)
    frequent_item_sets.to_csv(freq_file_path, sep='\t', index=False)
    # print(frequent_item_sets)
    rules = association_rules(frequent_item_sets, metric='lift', min_threshold=0.1)
    rules = rules[(rules['consequent support'] >=0.5)]
    rules.sort_values(by = 'lift',ascending=False,inplace=True)
    # print(rules)
    rules.to_csv(rule_file_path, sep='\t', index=False)
