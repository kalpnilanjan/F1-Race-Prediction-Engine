#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:37:07 2019

@author: admin
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


new_data_temp = pd.read_csv("Preprocessed _Data.csv")

final_sample_data = new_data_temp.iloc[:, [3, 4, 5, 6, 7, 8, 10,11, 12]]
X = final_sample_data
Y = new_data_temp['DriverPosition']
print("X",X.shape,"and Y:",Y.shape)

def baseline_model():
    model = Sequential()
    model.add(Dense(9, input_dim=9, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

print("Estimating")
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=False)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Accuracy: ", results.max()*100)