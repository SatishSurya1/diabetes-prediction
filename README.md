# diabetes-prediction
diabetes prediction
import numpy as np
from numpy import random
import pandas as pd
import os
from sklearn import linear_model, datasets, tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()
X = diabetes_dataset.drop(['Outcome'],axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

std = StandardScaler()
std_data = std.fit_transform(X)
print(std_data)

x = std_data
y = diabetes_dataset['Outcome']
print(X)
print(Y)

x_test = x[random_indices[537:-1]]
y_test = y[random_indices[537:-1]]

print(x_test)
print(y_test)

training the model

classifier = tree.DecisionTreeRegressor()

f = classifier.fit(X_train, y_train)
print(f)

model evaluation

X_train_pred = classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_pred, y_train)
print("accuracy scoring of the training data:",training_data_accuracy*100)

X_test_pred = classifier.predict(x_test)
test_data_accuracy = accuracy_score(X_test_pred,y_test)
print("accuracy score for test data:",test_data_accuracy*100)

input_data = (3,122,23,0,0,22.4,0.543,12)
inputdata_as_numpyarray = np.asarray(input_data)
data_reshape = inputdata_as_numpyarray.reshape(1,-1)
std_Data = std.transform(data_reshape)
print(std_Data)

prediction = classifier.predict(std_Data)
if prediction==0:
    print("no diabetes")
else:
    print("patient have diabetes")
