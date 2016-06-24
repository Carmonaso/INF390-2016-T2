# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:28:23 2016

@author: gotba_000
"""

import pandas as pd
import numpy as np

#A)

url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)

df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

#B)

print(df.shape)
df.info()
df.describe()

#C)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

#D)

import sklearn.linear_model as lm
X = df_scaled.ix[:,:-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
linreg = lm.LinearRegression(fit_intercept = False)
linreg.fit(Xtrain, ytrain)

#E)

import scipy as sc

weights = linreg.coef_
arounded_weights = np.around(weights,2)

std_errors = sc.stats.sem(Xtrain)[:8]
arounded_std_err = np.around(std_errors,2)

z_scores = []
for i in range(8):
    z_scores.append(round(weights[i] / std_errors[i],2))

#F)

from sklearn import cross_validation

def crossvalidation(K):
    Xm = Xtrain.as_matrix()
    ym = ytrain.as_matrix()
    k_fold = cross_validation.KFold(len(Xm),K)
    mse_cv = 0
    for k, (train, val) in enumerate(k_fold):
        linreg = lm.LinearRegression(fit_intercept = False)
        linreg.fit(Xm[train], ym[train])
        yhat_val = linreg.predict(Xm[val])
        mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
        mse_cv += mse_fold
    mse_cv = mse_cv / K
    return mse_cv

mse_cv_5 = crossvalidation(5);
mse_cv_10 = crossvalidation(10);

mse_test = np.mean((linreg.predict(Xtest) - ytest) ** 2)

#G)

import pylab
errores_train = (linreg.predict(Xtest) - ytest) ** 2
sc.stats.probplot(errores_train, plot = pylab)
pylab.show()