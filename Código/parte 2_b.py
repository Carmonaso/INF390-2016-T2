#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn import cross_validation
import matplotlib.pyplot as plt






def bss(x, y, names_x, k = 10000):
    p = x.shape[1]-1
    k = min(p, k)
    names_x = np.array(names_x)
    remaining = range(0, p)+[p]
    selected = remaining
    current_score = 0.0
    worst_new_score = 0.0


    while remaining and len(selected)>k :
        score_candidates = []

        for candidate in selected:
            selected_aux=[]
            for ind in selected:
                if (ind != candidate):
                    selected_aux=selected_aux+[ind]


            indexes = selected_aux
            x_train = x[:,indexes]
            model = lm.LinearRegression(fit_intercept=False)
            predictions_train = model.fit(x_train, y).predict(x_train)
            residuals_train = predictions_train - y
            mse_candidate = np.mean(np.power(residuals_train, 2))
            # print (mse_candidate, candidate)
            score_candidates.append((mse_candidate, candidate))

        # print " "



        score_candidates.sort(reverse=True)

        worst_new_score, worst_candidate = score_candidates.pop()
        selected.remove(worst_candidate)

    return selected






url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

x = df_scaled.ix[:,:-1]
N = x.shape[0]
x.insert(x.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']


names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]


y_train = y[istrain].as_matrix()
y_test = y[np.logical_not(istrain)].as_matrix()

eje_x = [1,2,3,4,5,6,7,8]

contador=1
testing=[]
entrenamiento=[]

#while contador > 0:
while contador < 9:
    x_train = x[istrain].as_matrix()
    predictores = bss(x_train,y_train,names_regressors,contador)
    x_train = x_train[:,predictores]
    model = lm.LinearRegression(fit_intercept=False)
    predictions_train = model.fit(x_train, y_train).predict(x_train)
    residuals_train = predictions_train - y_train
    mse_predictor = np.mean(np.power(residuals_train, 2))
    entrenamiento.append(mse_predictor)
    x_test = x[np.logical_not(istrain)].as_matrix()
    x_test = x_test[:,predictores]
    predictions_test = model.fit(x_train, y_train).predict(x_test)
    residuals_test = predictions_test - y_test
    mse_predictor_test = np.mean(np.power(residuals_test, 2))
    testing.append(mse_predictor_test)
    #contador=contador-1
    contador=contador+1



plt.plot(eje_x, testing, 'r-',color='r',label="Validacion")
plt.plot(eje_x, entrenamiento,'r-',color='b',label ="Entrenamiento")
plt.axis([9,0,0.3,1.5])
plt.xlabel('Numero Variables', fontsize=18)
plt.ylabel('Promedio Error Cuadratico', fontsize=16)
plt.legend()
plt.show()
