import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression, f_classif, SelectKBest


df = pd.read_csv("mirna.csv", delimiter=",", index_col=0)
print(df)
y = df.iloc[:, -1]
y
X = df.iloc[:, :-1]
X
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X)
X.shape
selector = SelectKBest(score_func=f_classif, k=100) 
X_new = selector.fit_transform(X, y)
selected_feature_names = X.columns[selector.get_support()]
selected_features_df = X[list(selected_feature_names)]
selected_features_df.to_csv('selected_features_1001.csv', index=True)
print("Indices of selected features:", selected_feature_names)



