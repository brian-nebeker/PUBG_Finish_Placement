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
from sklearn.model_selection import GridSearchCV


"""
DBNOs - Number of enemy players knocked.
assists - Number of enemy players this player damaged that were killed by teammates.
boosts - Number of boost items used.
damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
headshotKills - Number of enemy players killed with headshots.
heals - Number of healing items used.
Id - Player’s Id
killPlace - Ranking in match of number of enemy players killed.
killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
killStreaks - Max number of enemy players killed in a short amount of time.
kills - Number of enemy players killed.
longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
matchDuration - Duration of match in seconds.
matchId - ID to identify match. There are no matches that are in both the training and testing set.
matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
revives - Number of times this player revived teammates.
rideDistance - Total distance traveled in vehicles measured in meters.
roadKills - Number of kills while in a vehicle.
swimDistance - Total distance traveled by swimming measured in meters.
teamKills - Number of times this player killed a teammate.
vehicleDestroys - Number of vehicles destroyed.
walkDistance - Total distance traveled on foot measured in meters.
weaponsAcquired - Number of weapons picked up.
winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
numGroups - Number of groups we have data for in the match.
maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.
"""


#Feature Engineering
df = pfunc.load_df()
drop_columns = ['Id', 'groupId', 'matchId', 'matchType']

df['lobby_ct'] = df.groupby('matchId')['matchId'].transform('count')
df['perspective_code'] = df['matchType'].apply(lambda x: 0 if 'fpp' in x else 1)

df['total_heals'] = df['boosts'] + df['heals']
df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
df['kills_norm'] = df['kills']*(((100-df['lobby_ct'])/100)+1)
df['damageDealt_norm'] = df['damageDealt']*(((100-df['lobby_ct'])/100)+1)

df['headshotAcc'] = df['headshotKills'] / df['kills']
df['headshotAcc'].fillna(0, inplace=True)
df['headshotAcc'].replace(np.inf, 0, inplace=True)

df['kill_per_heal'] = df['kills'] / df['total_heals']
df['kill_per_heal'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['kill_per_heal'].fillna(df['kills'], inplace=True)

df['distance_per_heal'] = df['total_Distance'] / df['total_heals']
df['distance_per_heal'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['distance_per_heal'].fillna(df['total_Distance'], inplace=True)


# Rank features by their group within a mtach
non_rank_col = ['Id', 'groupId', 'matchId', 'matchType', 'matchDuration', 'numGroups', 
                'matchType_code', 'lobby_ct', 'perspective_code', 'winPlacePerc']
features = [c for c in df if c not in non_rank_col]

# Average and Rank by group
df = pfunc.avg_and_rank_by_group(df, features, 'min')
df = pfunc.avg_and_rank_by_group(df, features, 'max')
df = pfunc.avg_and_rank_by_group(df, features, 'median')
df = pfunc.avg_and_rank_by_group(df, features, 'mean')
df = pfunc.avg_and_rank_by_group(df, features, 'sum')

dummy = pd.get_dummies(df['matchType'])
df = pd.concat([df, dummy], axis=1)


# Remove colinearity
drop_columns = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
X_cols = [c for c in df if c not in drop_columns]
y = 'winPlacePerc'


# Decrease df size to train quicker for hypertuning
X_train, X_test, y_train, y_test = pfunc.train_test_split(df, X_cols, y, sub_set=0.4)



# Pipeline and GridSearch
#==============================================================================#
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Model for pipeline
model = MLPRegressor(hidden_layer_sizes=(250, 200, 150, 100),
                      activation='relu',
                      solver='adam',
                      max_iter=400,
                      alpha=0.0001
                      ) # -0.027064


# Pipeline with preprocessing and model
pipe = Pipeline([('preproc', preprocessing.StandardScaler()), ('model', model)])


# Parameters for gridsearch
parameters = {}

# Gridsearch and fit
print('Gridsearch Start')
search = GridSearchCV(pipe, param_grid=parameters, cv=3, scoring='neg_mean_absolute_error', n_jobs=5)

search.fit(X_train, y_train)
print('====================Total DF Results====================')
print(search.best_params_,'\n')
print(search.best_score_,'\n')


pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)


print(MAE) # 0.028295834875385204
