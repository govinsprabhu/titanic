# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:15:17 2018

@author: 609600403
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler



# Droping the unwanted set

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

combine[0] = combine[0].drop(['Name','Ticket', 'Cabin'], axis = 1)
combine[1] = combine[1].drop(['Name','Ticket', 'Cabin'], axis = 1)


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female': 1}).fillna(0).astype(str).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1,'Q':2}) .fillna(0).astype(int)   


# Split
X = combine[0].iloc[:, 2:12].values
y= combine[0].iloc[:, 1:2].values

x_test = combine[1].iloc[:, 1:12].values

# Encoding the categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 1] = labelencoder_X.fit_transform(X[:, 1])


# Replace the null values 

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
#imputer = imputer.fit(X[:, 0:6])
#X[:, 0:6] = imputer.transform(X[:, 0:6])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0:7])
X[:, 0:7] = imputer.transform(X[:, 0:7])


imputer = imputer.fit(x_test[:, 0:7])
x_test[:, 0:7] = imputer.transform(x_test[:, 0:7])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

x_test = sc.fit_transform(x_test)




from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained = pca.explained_variance_ratio_
x_test = pca.transform(x_test)
 





from sklearn.tree import DecisionTreeClassifier
random_forest = DecisionTreeClassifier()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest 

# 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(x_test)
classifier_acc = round(classifier.score(X_train, y_train) * 100, 2)
classifier_acc 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



submission = pd.DataFrame({
            "PassengerId":test_df["PassengerId"],
            "Survived" : y_pred
        })

submission.to_csv('./output/submission_normilized.csv', index = False)    

