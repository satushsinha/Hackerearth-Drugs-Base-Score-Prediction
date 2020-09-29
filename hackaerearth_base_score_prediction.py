# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:44:51 2020

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_set = pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')



X=train_set.iloc[:,2:3]
Y=test_set.iloc[:,5:6]
X=np.array(X)
Y=np.array(Y)
XplusY = np.concatenate((X, Y), axis = 0)
XplusY=pd.DataFrame(XplusY)
X=pd.DataFrame(X)
Y=pd.DataFrame(Y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
XplusY.iloc[:, 0]=labelencoder_X.fit_transform(XplusY.iloc[:, 0])




onehotencoder = OneHotEncoder(categorical_features = [0])
XplusY = onehotencoder.fit_transform(XplusY).toarray()




X.iloc[:, 0] = labelencoder_X.transform(X.iloc[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.transform(X).toarray()

Y.iloc[:, 0] = labelencoder_X.transform(Y.iloc[:, 0])
Y = onehotencoder.transform(Y).toarray()        

X1=train_set.iloc[:,3:4]
Y1=test_set.iloc[:,2:3]
X1=np.array(X1)
Y1=np.array(Y1)
X1plusY1 = np.concatenate((X1, Y1), axis = 0)
X1plusY1=pd.DataFrame(X1plusY1)


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


ignore_words = ['?', '!','#','.','...']
                
                

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 42925):
    review = re.sub('[^a-zA-Z]', ' ', X1plusY1[0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    if review not in ignore_words:
        corpus.append(lemmatizer.lemmatize(review.lower()))
        
        


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X1plusY1 = cv.fit_transform(corpus).toarray()

X1=X1plusY1[0:32165,:]
Y1=X1plusY1[32165:42925,:]

X2=train_set.iloc[:,[4,6]]
Y2=test_set.iloc[:,[6,4]]

X=np.array(X)

X1=np.array(X1)
X2=np.array(X2)
Y=np.array(Y)
Y1=np.array(Y1)
Y2=np.array(Y2)

X_train = np.concatenate((X, X1, X2,), axis = 1)
X_test = np.concatenate((Y, Y1, Y2,), axis = 1)

Y_train=train_set.iloc[:,7:8].values  

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_y.fit_transform(Y_train)
X_test=sc_X.transform(X_test)
'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, Y_train)
'''

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train)
        

y_pred = regressor.predict(X_test)

dataset= pd.read_csv('submission.csv')
final=test_set.iloc[:,0]


dataset['patient_id'] = final
dataset.to_csv("submission.csv", index=False)

dataset['base_score'] =y_pred
dataset.to_csv("submission.csv", index=False)

