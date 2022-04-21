import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def pricepredict():
    knn = KNeighborsClassifier()
    data_train = pd.read_csv(r'./train.csv')
    data_test = pd.read_csv(r'./test.csv')
    info = data_train.info()
    std = StandardScaler() 
    x = data_train.drop('price_range',axis=1)
    y = data_train['price_range']
    data_test = data_test.drop('id',axis=1)
    X_std = std.fit_transform(x)
    data_test_std = std.transform(data_test)
    knn.fit(X_std,y)
    prediction =knn.predict()
    print(prediction)
    return prediction

df = pd.read_csv("./data.csv")
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['Brand me'],axis=1,inplace=True)

df['Ratings'] = df['Ratings'].fillna(df['Ratings'].mean())
df['RAM'] = df['RAM'].fillna(df['RAM'].mean())
df['ROM'] = df['ROM'].fillna(df['ROM'].mean())
df['Mobile_Size'] = df['Mobile_Size'].fillna(df['Mobile_Size'].mean())
df['Selfi_Cam'] = df['Selfi_Cam'].fillna(df['Selfi_Cam'].mean())

df['RAM'] = df['RAM'].astype('int64')
df['ROM'] = df['ROM'].astype('int64')
df['Selfi_Cam'] = df['Selfi_Cam'].astype('int64')

corr = df.corr()

X = df.iloc[:,1:7]  # Independent columns
y = df.iloc[:,[-1]]

bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#model = ExtraTreesClassifier()
#model.fit(X,y)

#X = df.iloc[:,[6,2,4,5,1,3]]
#y = df.iloc[:,[-1]]

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)

#reg = RandomForestRegressor()
#reg.fit(X_train,y_train)

#y_pred = reg.predict(X_train)

#reg.predict([[4.0,128.0,6.00,48,13.0,4000]])