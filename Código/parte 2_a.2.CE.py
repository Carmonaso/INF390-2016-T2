#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn import cross_validation
import matplotlib.pyplot as plt






def fss(x, y, names_x, k = 10000):
    #Cantidad de dimensiones -1
    p = x.shape[1]-1
    k = min(p, k)
    #Nombre variables
    names_x = np.array(names_x)

    #Lista de variables.
    remaining = range(0, p)

    #Selec comienza con la última variable.
    selected = [p]
    current_score = 0.0
    best_new_score = 0.0

    #Mientras la cantidad de variables seleccionadas sea menor que K
    while remaining and len(selected)<=k :
        score_candidates = []
        for candidate in remaining:
            #Se agrega el modelo.
            model = lm.LinearRegression(fit_intercept=False)

            #Se agrega la variable candidata que vendra de la lista de variables de remaining.
            indexes = selected + [candidate]

            #Los datos de entrenamiento serán filas x cantidad de variables para fitear el modelo.
            x_train = x[:,indexes]
            predictions_train = model.fit(x_train, y).predict(x_train)

            #Se compara los resultados experimentales con los resultados teoricos.
            residuals_train = predictions_train - y

            #MSE será el promedio de las diferencias al cuadrado de los residuals_train
            mse_candidate = np.mean(np.abs(residuals_train))

            #Se agrega el score agregando la variable candidate
            # print (mse_candidate, candidate)
            score_candidates.append((mse_candidate, candidate))

        # print " "
        score_candidates.sort()
        score_candidates[:] = score_candidates[::-1]
        best_new_score, best_candidate = score_candidates.pop()
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        # print "selected = %s ..."%names_x[best_candidate]
        # print "totalvars=%d, mse = %f"%(len(indexes),best_new_score)
        # print " "

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


entrenamiento =[]
testing = []
eje_x = [1,2,3,4,5,6,7,8]

contador = 1
while contador < 9:

    x_train = x[istrain].as_matrix()
    predictores = fss(x_train,y_train,names_regressors,contador)
    print predictores
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

    mse_predictor_test = np.mean(np.abs(residuals_test))
    testing.append(mse_predictor_test)


    contador=contador+1



plt.plot(eje_x, testing, 'r-',color='r',label="Validacion")
plt.plot(eje_x, entrenamiento,'r-',color='b',label ="Entrenamiento")
plt.axis([0,9,0.3,1.5])
plt.xlabel('Numero Variables', fontsize=18)
plt.ylabel('Promedio Error Absoluto', fontsize=16)
plt.legend()
plt.show()
