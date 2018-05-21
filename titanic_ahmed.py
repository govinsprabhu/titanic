# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:39:33 2018

@author: 609600403
"""

# https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category= DeprecationWarning)


import pandas as pd
pd.options.display.max_columns= 100

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns

import pylab as plot
params = {
        'axes.labelsize' :"large",
        'xtick.labelsize' : "x-large",
        'legend.fontsize' : 20,
        'figure.dpi' : 150,
        'figure.figsize' : [25, 7]
        }

plot.rcParams.update(params) 

data = pd.read_csv('./data/train.csv')

data.shape

data.head()

data.describe()

data['Age'] = data['Age'].fillna(data['Age'].median())


data.describe()

data['Died'] = 1 - data['Survived']

data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind = 'bar', figsize = (, 0.5), stacked = True, colors = ['g', 'r'])

figure = plt.figure(figsize = (2,0.5))
