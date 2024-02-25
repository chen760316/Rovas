# unsupervised methods
from deepod.models import REPEN, SLAD, ICL, NeuTraL
from deepod.models.tabular import GOAD
from deepod.models.tabular import RCA
from deepod.models.tabular import DeepSVDD
from deepod.models import DeepSAD, RoSAS, PReNet
from deepod.models.tabular import DevNet
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
"""
kaggle datasets
"""
file_path = "../../kaggle_datasets/Apple_Quality/apple_quality.csv"
apple_quality = pd.read_csv(file_path)
enc = LabelEncoder()
apple_quality['Quality'] = enc.fit_transform(apple_quality['Quality'])
X = apple_quality.values[:,1:8]
y = apple_quality.values[:,8]
X = StandardScaler().fit_transform(X)
indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.5, random_state=1)
    
"""GOAD"""
clf_1 = GOAD(epochs=epochs, device=device, n_trans=n_trans)
clf_1.fit(X_train, y=None)
preds_1 = clf_1.predict(X_test)
"""DeepSVDD"""
clf_2 = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
clf_2.fit(X_train, y=None)
preds_2 = clf_2.predict(X_test)
"""RCA"""
clf_3 = RCA(epochs=epochs, device=device, act='LeakyReLU')
clf_3.fit(X_train)
preds_3 = clf_3.predict(X_test)
"""RePEN"""
clf_4 = REPEN(epochs=5, device=device)
clf_4.fit(X_train)
preds_4 = clf_4.predict(X_test)
"""SLAD"""
clf_5 = SLAD(epochs=2, device=device)
clf_5.fit(X_train)
preds_5 = clf_5.predict(X_test)
"""ICL"""
clf_6 = ICL(epochs=1, device=device, n_ensemble='auto')
clf_6.fit(X_train)
preds_6 = clf_6.predict(X_test)
"""NeuTraL"""
clf_7 = NeuTraL(epochs=1, device=device)
clf_7.fit(X_train)
preds_7 = clf_7.predict(X_test)
"""DevNet"""
clf_8 = DevNet(epochs=1, hidden_dims=20, device=device,
                          random_state=42)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf_8.fit(X_train, y_semi)
preds_8 = clf_8.predict(X_test)
"""DeepSAD"""
clf_9 = DeepSAD(epochs=1, hidden_dims=20,
                   device=device,
                   random_state=42)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf_9.fit(X_train, y_semi)
preds_9 = clf_9.predict(X_test)
"""RoSAS"""
clf_10 = RoSAS(epochs=1, hidden_dims=20, device=device, random_state=42)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf_10.fit(X_train, y_semi)
preds_10 = clf_10.predict(X_test)
"""PReNet"""
clf_11 = PReNet(epochs=1,
                  epoch_steps=20,
                  device=device,
                  batch_size=256,
                  lr=1e-5)
anom_id = np.where(y_train == 1)[0]
known_anom_id = np.random.choice(anom_id, 10, replace=False)
y_semi = np.zeros_like(y_train, dtype=int)
y_semi[known_anom_id] = 1
clf_11.fit(X_train, y_semi)
preds_11 = clf_11.predict(X_test)

"""
四分位极差法
"""
test_data = apple_quality.iloc[idx_test]
q1 = test_data['Size'].quantile(0.25)
q3 = test_data['Size'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

preds_12 = [1 if x > upper_bound or x < lower_bound else 0 for x in test_data['Size']]
"""
Z-分数法
"""
test_data = apple_quality.iloc[idx_test]
# test_data = test_data[test_data['Nutrition_Status'] == target_class]
test_data['Size_zscore'] = stats.zscore(test_data['Size'])
preds_13 = [1 if x > 3 or x < -3 else 0 for x in test_data['Size_zscore']]


svm_model = svm.SVC(C=10)
# svm_model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)
result_list = [1 if x == y else 0 for x, y in zip(preds, y_test)]
dataframe = pd.DataFrame({'GOAD':preds_1, 'DeepSVDD':preds_2, 'RCA':preds_3, 'RePEN':preds_4, 'SLAD':preds_5, 'ICL':preds_6, 'NeuTraL':preds_7,\
                           'DevNet':preds_8, 'DeepSAD':preds_9, 'RoSAS':preds_10, 'PReNet':preds_11, 'Quartile':preds_12, 'Z_score':preds_13, 'label':y_test, 'Success':result_list})
dataframe.to_csv("apple_rule.csv",index=False,sep=',')
