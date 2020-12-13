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

import matplotlib.pyplot as plt
import seaborn as sns


# Read training csv
df = pfunc.load_df()

# Create Lobby Features
df['lobby_ct'] = df.groupby('matchId')['matchId'].transform('count')

# Codify matchType to regular=0, event=1, custom=2
regular_matches = ['duo', 'duo-fpp', 'solo', 'solo-fpp', 'squad', 'squad-fpp']
event_matches = ['crashfpp', 'crashtpp', 'flarefpp', 'flaretpp']
df['matchType_code'] = df['matchType'].apply(lambda x: 0 if x in regular_matches else 1 if x in event_matches else 2)

# Create Individual Features
df['total_heals'] = df['boosts'] + df['heals']
df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']

# How many kills were done with headshots
df['headshotAcc'] = df['headshotKills'] / df['kills']
df['headshotAcc'].replace([np.inf, -np.inf], 0, inplace=True)
df['headshotAcc'].fillna(df['headshotKills'], inplace=True)

# Roadkills without any driving
df['roadKills_per_distance'] = df['roadKills'] / df['rideDistance']
df['roadKills_per_distance'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['roadKills_per_distance'].fillna(df['roadKills'], inplace=True)
# df['roadKills_per_distance'] = df['roadKills_per_distance'].apply(lambda x: x if x <= 1.5 else 2)

# Kills with nearly 0 movement
df['kill_per_distance'] = df['kills'] / df['total_Distance']
df['kill_per_distance'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['kill_per_distance'].fillna(df['kills'], inplace=True)
# df['kill_per_distance'] = df['kill_per_distance'].apply(lambda x: x if x <= 1.5 else 2)



cheater_groups = []
cheater_groups = cheater_groups + df[(df['headshotAcc']==1) & (df['matchType_code']<2) & (df['kills']>4)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['roadKills_per_distance']>1) & (df['matchType_code']<2) & (df['roadKills']>1)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['kill_per_distance']>1) & (df['matchType_code']<2) & (df['kills']>1)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['kills']>25) & (df['matchType_code']<2)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['longestKill']>=1000) & (df['matchType_code']<2)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['weaponsAcquired']>=100) & (df['matchType_code']<2)]['groupId'].tolist()
cheater_groups = cheater_groups + df[(df['total_heals']>=40) & (df['matchType_code']<2)]['groupId'].tolist()



cheaters = df[df['groupId'].isin(cheater_groups)]

