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
import matplotlib.pyplot as plt
import PUBG_functions as pfunc

import shap

from sklearn.pipeline import Pipeline
import sklearn.preprocessing as pp
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor



# Read training csv
df = pfunc.load_df()


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


# Individual experiments
experiments = [pfunc.original, pfunc.lobby_ct, pfunc.total_kills, pfunc.matchType_code,
               pfunc.perspective_code,
               pfunc.total_heals, pfunc.kills_norm, pfunc.damageDelt_norm, pfunc.headshotAcc, 
               pfunc.total_distance, pfunc.distance_per_kill, pfunc.kill_per_distance, 
               pfunc.heal_per_kill, pfunc.kill_per_heal, pfunc.heal_per_distance, 
               pfunc.distance_per_heal, pfunc.portion_of_kills,
               pfunc.group_killPlace_avg, pfunc.min_by_group, pfunc.max_by_group, 
               pfunc.sum_by_group, pfunc.median_by_group, pfunc.mean_by_group, 
               pfunc.rank_by_group]


# Filters for df in experiments
# match_contains = {'All':('',True), 'Squad':('squad',True), 'Duo':('duo',True), 'Solo':('solo',True), 'Other':('solo|duo|squad',False)}
match_contains = {'All':('',True), 'Single':('solo',True), 'Multi':('solo',False)}

full_results = pd.DataFrame(data=[e.__name__ for e in experiments], columns=['name'])

for n, t in match_contains.items():
    results = pfunc.run_experiments_lr(experiments, t[0], t[1])
    full_results = full_results.merge(results, on='name', how='left', suffixes=['','_'+n])

full_results = full_results.set_index('name', drop=True)

results_difference = full_results.subtract(full_results.loc['original'], axis=1) * -1

N = 24
ind = np.arange(N)
width = 0.25

plt.figure(figsize=(22,8))

for x in range(0,3):
    height = results_difference.iloc[:, x]
    label = label=results_difference.columns.values[x]
    plt.bar(ind + ((x)*width), height, width, label=label, align='center')


plt.xticks(ind + width, labels=results_difference.index.tolist(), rotation=270)
plt.legend(loc='best')
plt.hlines(0, 0, 24)
plt.show()









from sklearn.tree import DecisionTreeRegressor

temp = df.copy()

y = 'winPlacePerc'
drop_cols = ['Id', 'groupId', 'matchId', 'matchType', y]
X_cols = [c for c in temp if c not in drop_cols]

train, test = pfunc.train_test_split(temp)

model = DecisionTreeRegressor()
model.fit(train[X_cols], train[y])

importances = model.feature_importances_

imp_num = []
imp_score = []

for i,v in enumerate(importances):
    imp_num.append(i)
    imp_score.append(v)
    print('Feature: %0d, Score: %.5f' % (i,v))

imp_df = pd.DataFrame(list(zip(imp_num, X_cols, imp_score)), columns=['number', 'feature', 'score'])
imp_df.sort_values(by='score', inplace=True)


# shap.initjs()
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(train[X_cols])
# shap.summary_plot(shap_values, train[X_cols], plot_type='bar')
# shap.summary_plot(shap_values, train[X_cols], feature_names=X_cols)


imp_df.to_csv('RandomForestRegressor_importance.csv')

