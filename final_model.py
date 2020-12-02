# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:01:01 2020

@author: bnebe
"""
import os
import sys

def add_module_path_to_system():
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        return module_path 

module_path = add_module_path_to_system()

import numpy as np
import pandas as pd
import random
import PUBG_functions as pfunc

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor


#Feature Engineering
df_train = pfunc.load_df()
df_train = pfunc.preproc_df(df_train)

# Create X and y columns
drop_columns = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
X_cols = [c for c in df_train if c not in drop_columns]
y = 'winPlacePerc'

# Split df to train
X_train = df_train[X_cols]
y_train = df_train[y]

X_train = X_train.copy()
y_train = y_train.copy()


# LGBM Model
model_lgbm = lgb.LGBMRegressor(boosting_type = 'gbdt',
                               learning_rate = 0.05,
                               objective= 'mae',
                               metric = 'mae',
                               num_leaves = 900,
                               verbose = 1,
                               n_estimators=10000)

# Pipeline with preprocessing and model
pipe_lgbm = Pipeline([('preproc', preprocessing.StandardScaler()), ('model', model_lgbm)])



# MLP Model
model_mlp = MLPRegressor(hidden_layer_sizes=(250, 200, 150, 100),
                         activation='relu',
                         solver='adam',
                         max_iter=400,
                         alpha=0.0001)

# Pipeline with preprocessing and model
pipe_mlp = Pipeline([('preproc', preprocessing.StandardScaler()), ('model', model_mlp)])


# Fit training data to models
pipe_lgbm.fit(X_train, y_train)
pipe_mlp.fit(X_train, y_train)

# Delete training data to save memory
del df_train, X_train, y_train


# Load and Feature Engineer test data
df_test = pd.read_csv('test.csv')
df_test = pfunc.preproc_df(df_test)

# Split df to test/predict
X_test = df_test[X_cols]
X_test = X_test.copy()


# Predict with lgbm and save results as csv
y_pred_lgbm = pipe_lgbm.predict(X_test)


# Predict with mlp and save results as csv
y_pred_mlp = pipe_mlp.predict(X_test)


def adjust_winPlacePerc(df, column):
    df_sub_group = df.groupby(['matchId', 'groupId']).first().reset_index()
    df_sub_group['rank'] = df_sub_group.groupby(['matchId'])[column].rank()
    df_sub_group = df_sub_group.merge(df_sub_group.groupby('matchId')['rank'].max().to_frame('max_rank').reset_index(), on='matchId', how='left')
    df_sub_group['adjusted_perc'] = (df_sub_group['rank'] - 1) / (df_sub_group['numGroups'] - 1)
    
    df = df.merge(df_sub_group[['adjusted_perc', 'matchId', 'groupId']], on=['matchId', 'groupId'], how='left')
    df["winPlacePerc"] = df["adjusted_perc"]
    df['winPlacePerc']
    return df


df_final = df_test.copy()
df_final['lgbm_pred'] = y_pred_lgbm
df_final['mlp_pred'] = y_pred_mlp
df_final = df_final[['Id', 'groupId', 'matchId', 'numGroups', 'lgbm_pred', 'mlp_pred']]
df_final = df_final.copy()
df_final.to_csv('df_final.csv')


df_final['weighted_perc'] = df_final['lgbm_pred']*0.5 + df_final['mlp_pred']*0.5

test = adjust_winPlacePerc(df_final, 'weighted_perc')

temp = test[test['winPlacePerc'].isna()]

