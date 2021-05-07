# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:21:22 2020

@author: Salmaan Ahmed Ansari
"""



# Importing the libraries
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
dataset = pd.read_csv('train.csv', sep = ',')
dataset.describe()
dataset.info()

X = dataset.iloc[:, [1, 2, 3, 5, 6, 7, 8, 9, 10]].values

y = dataset.iloc[:, 11].values

plt.figure(figsize=(11,11))
sns.heatmap(dataset.corr())

"""
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, [0,1,3,4,6]])
X[:, [0,1,3,4,6]] = imputer.transform(X[:, [0,1,3,4,6]])
print(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [2,5]])
X[:, [2,5]] = imputer.transform(X[:, [2,5]])
print(X)"""


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 5] = le.fit_transform(X[:, 5])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)





"""
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000], 'kernel': ['rbf']},
              {'C': [1, 1.5, 2, 3, 4, 5, 6], 'kernel': ['rbf'], 'gamma': [0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
"""





#for test dataset

dataset_test = pd.read_csv('test.csv')
X_tes = dataset_test.iloc[:, 1:11].values



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_tes[:, 2] = le.fit_transform(X_tes[:, 2])
X_tes[:, 7] = le.fit_transform(X_tes[:, 7])





# Predicting the Test set results
y_tes = classifier.predict(X_tes)

y_tes



