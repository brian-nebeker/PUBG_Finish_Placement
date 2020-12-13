# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:33:54 2020

@author: bnebe
"""
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def load_df(match_contains='', contains_bool=True):
    df = pd.read_csv('train.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    
    df = df[df['matchType'].str.contains(match_contains)==contains_bool]
    
    return df

def train_test_split(df, cols, y, test_size=0.2, sub_set=1):
    if sub_set < 1:
        match_ids = df['matchId'].unique().tolist()
        train_size = int(len(match_ids) * (0.4))
        train_match_ids = random.sample(match_ids, train_size)
        df = df[df['matchId'].isin(train_match_ids)]
    
    
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1-test_size))
    train_match_ids = random.sample(match_ids, train_size)
    
    train = df[df['matchId'].isin(train_match_ids)]
    test = df[-df['matchId'].isin(train_match_ids)]
    
    X_train = train[cols]
    y_train = train[y]
    X_test = test[cols]
    y_test = test[y]
    
    return X_train, X_test, y_train, y_test


def run_process(process, match_contains='', contains_bool=True):
    print(process.__name__)
    df = load_df(match_contains, contains_bool)
    
    df = process(df)
    
    y = 'winPlacePerc'
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', y]
    X_cols = [c for c in df if c not in drop_cols]
    
    X_train, X_test, y_train, y_true = train_test_split(df, X_cols, y)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return mean_absolute_error(y_true, y_pred)

def run_experiments_lr(experiments, match_contains='', contains_bool=True):
    results = []
    
    for exp in experiments:
        score = run_process(exp, match_contains, contains_bool)
        results.append({'name':exp.__name__,
                        'score': score})
    
    return pd.DataFrame(results, columns=['name', 'score']).sort_values(by='score')

def run_experiments_rf(experiments):
    results = []
    
    for exp in experiments:
        score = run_process(exp)
        results.append({'name':exp.__name__,
                        'score': score})
    
    return pd.DataFrame(results, columns=['name', 'score']).sort_values(by='score')

def rank_df_by_features(df, features):
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    df = df.merge(agg, suffixes=['', '_rank'], how='left', on=['matchId', 'groupId'])
    return df


def preproc_df(df):
    #Feature Engineering
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
    df = avg_and_rank_by_group(df, features, 'min')
    df = avg_and_rank_by_group(df, features, 'max')
    df = avg_and_rank_by_group(df, features, 'median')
    df = avg_and_rank_by_group(df, features, 'mean')
    df = avg_and_rank_by_group(df, features, 'sum')
    
    dummy = pd.get_dummies(df['matchType'])
    df = pd.concat([df, dummy], axis=1)
    
    return df


# Features for experiments
def original(df):
    return df

#==============================================================================#

# Lobby Features
def lobby_ct(df):
    df['lobby_ct'] = df.groupby('matchId')['matchId'].transform('count')
    return df

def total_kills(df):
    df['total_kills'] = df.groupby('matchId')['kills'].transform('sum')
    return df

def matchType_code(df):
    regular_matches = ['duo', 'duo-fpp', 'solo', 'solo-fpp', 'squad', 'squad-fpp']
    event_matches = ['crashfpp', 'crashtpp', 'flarefpp', 'flaretpp']
    df['matchType_code'] = df['matchType'].apply(lambda x: 0 if x in regular_matches else 1 if x in event_matches else 2)
    return df

def perspective_code(df):
    df['perspective_code'] = df['matchType'].apply(lambda x: 0 if 'fpp' in x else 1)
    return df

#==============================================================================#

# Individual Features
def total_heals(df):
    df['total_heals'] = df['boosts'] + df['heals']
    return df

def kills_norm(df):
    df['lobby_ct'] = df.groupby('matchId')['matchId'].transform('count')
    df['kills_norm'] = df['kills']*(((100-df['lobby_ct'])/100)+1)
    df = df.drop(['lobby_ct'], axis=1)
    return df

def damageDelt_norm(df):
    df['lobby_ct'] = df.groupby('matchId')['matchId'].transform('count')
    df['damageDealt_norm'] = df['damageDealt']*(((100-df['lobby_ct'])/100)+1)
    df = df.drop(['lobby_ct'], axis=1)
    return df

def headshotAcc(df):
    df['headshotAcc'] = df['headshotKills'] / df['kills']
    df['headshotAcc'].fillna(0, inplace=True)
    df['headshotAcc'].replace(np.inf, 0, inplace=True)
    return df
    
def total_distance(df):
    df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    return df

def distance_per_kill(df):
    df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    
    df['distance_per_kill'] = df['total_Distance'] / df['kills']
    df['distance_per_kill'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['distance_per_kill'].fillna(df['total_Distance'], inplace=True)
    
    df = df.drop(['total_Distance'], axis=1)
    return df

def kill_per_distance(df):
    df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    
    df['kill_per_distance'] = df['kills'] / df['total_Distance']
    df['kill_per_distance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['kill_per_distance'].fillna(df['kills'], inplace=True)
    
    df = df.drop(['total_Distance'], axis=1)
    return df

def distance_per_heal(df):
    df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    df['total_heals'] = df['boosts'] + df['heals']
    
    df['distance_per_heal'] = df['total_Distance'] / df['total_heals']
    df['distance_per_heal'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['distance_per_heal'].fillna(df['total_Distance'], inplace=True)
    
    df = df.drop(['total_Distance', 'total_heals'], axis=1)
    return df

def heal_per_distance(df):
    df['total_Distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    df['total_heals'] = df['boosts'] + df['heals']
    
    df['heal_per_distance'] = df['total_heals'] / df['total_Distance']
    df['heal_per_distance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['heal_per_distance'].fillna(df['total_heals'], inplace=True)
    
    df = df.drop(['total_Distance', 'total_heals'], axis=1)
    return df

def heal_per_kill(df):
    df['total_heals'] = df['boosts'] + df['heals']
    
    df['heal_per_kill'] = df['total_heals'] / df['kills']
    df['heal_per_kill'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['heal_per_kill'].fillna(df['total_heals'], inplace=True)
    
    df = df.drop(['total_heals'], axis=1)
    return df

def kill_per_heal(df):
    df['total_heals'] = df['boosts'] + df['heals']
    
    df['kill_per_heal'] = df['kills'] / df['total_heals']
    df['kill_per_heal'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['kill_per_heal'].fillna(df['kills'], inplace=True)
    
    df = df.drop(['total_heals'], axis=1)
    return df

def portion_of_kills(df):
    df['total_kills'] = df.groupby('matchId')['kills'].transform('sum')
    
    df['portion_of_kills'] = df['kills'] / df['total_kills']
    df['portion_of_kills'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['portion_of_kills'].fillna(0, inplace=True)
    
    df = df.drop(columns=['total_kills'], axis=1)
    return df

#==============================================================================#

# Group Features
def group_ct(df):
    agg = df.groupby(['groupId']).size().reset_index(name='group_ct')
    df = df.merge(agg, how='left', on=['groupId'])
    return df

def group_killPlace_avg(df):
    df['group_killPlace_avg'] = df.groupby('groupId')['killPlace'].transform('mean')
    return df

def min_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_group(df):
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [c for c in df if c not in drop_cols]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_rank'], how='left', on=['matchId', 'groupId'])

def avg_and_rank_by_group(df, features, function):
    agg = df.groupby(['matchId', 'groupId'])[features].agg(function)
    agg2 = agg.groupby('matchId')[features].rank(pct=True)
    agg = agg.join(agg2, lsuffix='_' + function, rsuffix='_' + function + '_rank')
    df = df.merge(agg, suffixes=['', ''], how='left', on=['matchId', 'groupId'])
    return df

#==============================================================================#

# Find cheaters in dataset
def cheater_groups(df):
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
    
    return cheater_groups