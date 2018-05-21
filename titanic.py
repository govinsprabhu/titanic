# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:48:07 2018

@author: 609600403
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

combine = [train_df, test_df]

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by= 'Survived', ascending = False)

g = sns.FacetGrid(train_df, row = 'Survived')
g.map(plt.hist, 'Age', bins= 20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


grid = sns.FacetGrid(train_df, row = 'Embarked', size =2.2, aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette = 'deep')
grid.add_legend()


grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size=2.2, aspect =1.6 )
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)

print ('Before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
print ('After', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

combine = [train_df, test_df]

for dataset in combine:
    dataset['title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

pd.crosstab(train_df['title'], train_df['Sex'])

for dataset in combine:
    dataset['title'] = dataset['title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['title'] = dataset['title'].replace('Mile', 'Miss')
    dataset['title'] = dataset['title'].replace('Ms', 'Miss')
    dataset['title'] = dataset['title'].replace('Mme', 'Mrs')

train_df[['title', 'Survived']].groupby(['title'], as_index = False).mean()
    
title_mapping = {"Mr": 1, "Miss": 2, "Mrs" : 3, "Master" : 4, "Rare": 5}
for dataset in combine:
    dataset['title'] = dataset['title'].map(title_mapping)
    dataset['title'] = dataset['title'].fillna(0)

dataset['Sex']
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(str).astype(int)


train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)

combine = [train_df, test_df]


test_df.head()
guess_ages = np.zeros((2,3))


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)            
train_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by = 'AgeBand', ascending = True)           

train_df = train_df.drop(['AgeBand'], axis = 1)            
            
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()    
                 
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)    

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    
train_df[['IsAlone', 'Survived']].groupby(['Survived'], as_index = False).mean()
            
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)

combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
            
            
test_df['Age*Class'] = test_df.Age * test_df.Pclass



for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port).map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) 
           
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)           
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace = True)           
            
            
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by = 'FareBand', ascending = True)            
            
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) , 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
            
train_df = train_df.drop(['FareBand'], axis = 1)
combine = [train_df, test_df]        
test_df.head(10)    

X_train = train_df.drop("Survived", axis = 1)
y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis = 1).copy()

X_train.shape, y_train.shape, X_test.shape            

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#Logical Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)

#Regression coeficient 
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by = 'Correlation', ascending = False)

# Support Vectore Mahines 
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc



# K- Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, y_train)
y_preed = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acc_percpetron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_percpetron


# Linear SVC
linearSVC = LinearSVC()
linearSVC.fit(X_train, y_train)
y_pred = linearSVC.predict(X_test)
acc_linear_svc = round(linearSVC.score(X_train, y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradiant Decent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train,y_train)  * 100, 2)


# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# Random Forest 
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_percpetron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
         "PassengerId" :test_df["PassengerId"],
         "Survived" :y_pred
        })
    
submission.to_csv('./output/submission.csv', index=False)    



















         
            
            
            
            
            
            
            
            
            
            
            
    