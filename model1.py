# Importing the libraries:
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
import json
from sklearn import preprocessing

# Importing dataset:
    
train = pd.read_excel(r"C:\Users\bhara\Downloads\salarydata.csv")
X = train.drop('salary', axis=1)
y = train['salary']

# Splitting the dataset to Training set and Test set:

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# fitting GradientBoostingClassifier to Training set

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)
gbc.fit(X_train, y_train)
# preditting the Test set results

ypred = gbc.predict(X_test)


# Saving model using pickle
pickle.dump(gbc,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

