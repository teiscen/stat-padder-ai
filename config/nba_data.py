from util.CSV_Data import CSV_Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import os

STARTING_FOLDER       = os.path.join('.', '_Data', 'NBA', 'Roster')
PLAYERS_CSV_FILE      = os.path.join(STARTING_FOLDER, 'players.csv')
PARTICIPATED_CSV_FILE = os.path.join(STARTING_FOLDER, 'participatedCSV.csv')
GAMES_CSV_FILE        = os.path.join(STARTING_FOLDER, 'gameCSV.csv')

FEATURE_COLS = [
    'Minutes', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS',
    'FG_successes', 'FG_attempts', 'FT_successes', 'FT_attempts',
    'ThreePT_successes', 'ThreePT_attempts'
]
CONTEXT_COLS = ['playerID', 'awayTeamID', 'isHome']
TARGET_COLS  = [
    'Fantasy_Pts', 'ThreePT_successes', 'FG_successes', 'FT_successes',
    'REB', 'AST', 'BLK', 'STL', 'TO'
]

SEQUENCE_LENGTH = 10
BATCH_SIZE      = 32

# Helper methods to format the data
def _split_baskets(csv_data):
    for column in ['FG', 'FT', 'ThreePT']:
        csv_data[[f'{column}_successes', f'{column}_attempts']] = csv_data[column].str.split('-', expand=True).astype(int)
        csv_data.drop([column], axis=1, inplace=True)

def _split_awayTeamID(nba_data):
    nba_data['isHome'] = (~nba_data['awayTeamID'].str.startswith('@')).astype(int)
    nba_data['awayTeamID'] = (
        nba_data['awayTeamID']
        .str.replace('@', '', regex=False)
        .str.replace('vs', '', regex=False)
        .str.strip()
        .str.lower()
    )

def _encode_categoricals(csv_data):
    for col in ['playerID', 'teamID', 'awayTeamID', 'position']:
        le = LabelEncoder()
        csv_data[col] = le.fit_transform(csv_data[col].astype(str))
    return csv_data

def _add_fantasy_points(csv_data):
    csv_data['Fantasy_Pts'] = (
        csv_data['ThreePT_successes'] * 3   +
        csv_data['FG_successes']      * 2   +
        csv_data['FT_successes']      * 1   +
        csv_data['REB']               * 1.2 +
        csv_data['AST']               * 1.5 +
        csv_data['BLK']               * 2   +
        csv_data['STL']               * 2   +
        csv_data['TO']                * -1
    )

def _normalize_features(csv_data):
    scaler = StandardScaler()
    csv_data[FEATURE_COLS] = scaler.fit_transform(csv_data[FEATURE_COLS])
    return scaler

# Builder
def build_data():
    csv_list = [
        CSV_Data(PLAYERS_CSV_FILE,      ['playerName', 'playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']),
        CSV_Data(PARTICIPATED_CSV_FILE, ['FGPercent', 'ThreePTPercent', 'FTPercent'], _split_baskets),
        CSV_Data(GAMES_CSV_FILE,        ['result'], _split_awayTeamID)
    ]
    merge_list = [['playerID'], ['teamID', 'gameDate']]
    merged = CSV_Data.merge_csv_list(csv_list, merge_list)
    merged = merged.sort_values(["playerID", "gameDate"])
    _add_fantasy_points(merged)
    _encode_categoricals(merged)
    return merged

def _window_generator(arr, feature_idx, context_idx, target_idx,
                       player_idx, posn_idx, team_idx, opp_idx,
                       ctx_player_idx, ctx_opp_idx, ishome_idx):
    for i in range(len(arr) - SEQUENCE_LENGTH):
        sequence_inputs = {
            'input_player': arr[i : i + SEQUENCE_LENGTH, player_idx].astype(np.int32),
            'input_posn':   arr[i : i + SEQUENCE_LENGTH, posn_idx].astype(np.int32),
            'input_team':   arr[i : i + SEQUENCE_LENGTH, team_idx].astype(np.int32),
            'input_opp':    arr[i : i + SEQUENCE_LENGTH, opp_idx].astype(np.int32),
            'input_stats':  arr[i : i + SEQUENCE_LENGTH, feature_idx],
            'input_ctx_player': arr[i + SEQUENCE_LENGTH, ctx_player_idx].reshape(1).astype(np.int32),
            'input_ctx_opp':    arr[i + SEQUENCE_LENGTH, ctx_opp_idx].reshape(1).astype(np.int32),
            'input_isHome':     arr[i + SEQUENCE_LENGTH, ishome_idx].reshape(1),
        }
        targets = {
            'Fantasy_pts_output':       arr[i + SEQUENCE_LENGTH, target_idx[0]],
            'ThreePT_successes_output': arr[i + SEQUENCE_LENGTH, target_idx[1]],
            'FG_successes_output':      arr[i + SEQUENCE_LENGTH, target_idx[2]],
            'FT_successes_output':      arr[i + SEQUENCE_LENGTH, target_idx[3]],
            'REB_output':               arr[i + SEQUENCE_LENGTH, target_idx[4]],
            'AST_output':               arr[i + SEQUENCE_LENGTH, target_idx[5]],
            'BLK_output':               arr[i + SEQUENCE_LENGTH, target_idx[6]],
            'STL_output':               arr[i + SEQUENCE_LENGTH, target_idx[7]],
            'TO_output':                arr[i + SEQUENCE_LENGTH, target_idx[8]],
        }
        yield sequence_inputs, targets

def generate_sequences(build=False, csv_file=None):
    if build:
        csv_data = build_data()
    else:
        csv_data = CSV_Data(csv_file, []).getData()
        _encode_categoricals(csv_data)

    scaler = _normalize_features(csv_data)

    all_cols = FEATURE_COLS + CONTEXT_COLS + TARGET_COLS + ['position', 'teamID']

    feature_idx    = [all_cols.index(c) for c in FEATURE_COLS]
    context_idx    = [all_cols.index(c) for c in CONTEXT_COLS]
    target_idx     = [all_cols.index(c) for c in TARGET_COLS]
    player_idx     = all_cols.index('playerID')
    posn_idx       = all_cols.index('position')
    team_idx       = all_cols.index('teamID')
    opp_idx        = all_cols.index('awayTeamID')
    ctx_player_idx = all_cols.index('playerID')
    ctx_opp_idx    = all_cols.index('awayTeamID')
    ishome_idx     = all_cols.index('isHome')

    datasets = []
    for _, group in csv_data.groupby("playerID"):
        arr = group[all_cols].to_numpy(dtype=np.float32)

        if len(arr) <= SEQUENCE_LENGTH:
            continue

        ds = tf.data.Dataset.from_generator(
            generator=lambda a=arr: _window_generator(
                a, feature_idx, context_idx, target_idx,
                player_idx, posn_idx, team_idx, opp_idx,
                ctx_player_idx, ctx_opp_idx, ishome_idx
            ),
            output_signature=(
                {
                    'input_player':     tf.TensorSpec(shape=(SEQUENCE_LENGTH,),                    dtype=tf.int32),
                    'input_posn':       tf.TensorSpec(shape=(SEQUENCE_LENGTH,),                    dtype=tf.int32),
                    'input_team':       tf.TensorSpec(shape=(SEQUENCE_LENGTH,),                    dtype=tf.int32),
                    'input_opp':        tf.TensorSpec(shape=(SEQUENCE_LENGTH,),                    dtype=tf.int32),
                    'input_stats':      tf.TensorSpec(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS)), dtype=tf.float32),
                    'input_ctx_player': tf.TensorSpec(shape=(1,),                                  dtype=tf.int32),
                    'input_ctx_opp':    tf.TensorSpec(shape=(1,),                                  dtype=tf.int32),
                    'input_isHome':     tf.TensorSpec(shape=(1,),                                  dtype=tf.float32),
                },
                {
                    'Fantasy_pts_output':       tf.TensorSpec(shape=(), dtype=tf.float32),
                    'ThreePT_successes_output': tf.TensorSpec(shape=(), dtype=tf.float32),
                    'FG_successes_output':      tf.TensorSpec(shape=(), dtype=tf.float32),
                    'FT_successes_output':      tf.TensorSpec(shape=(), dtype=tf.float32),
                    'REB_output':               tf.TensorSpec(shape=(), dtype=tf.float32),
                    'AST_output':               tf.TensorSpec(shape=(), dtype=tf.float32),
                    'BLK_output':               tf.TensorSpec(shape=(), dtype=tf.float32),
                    'STL_output':               tf.TensorSpec(shape=(), dtype=tf.float32),
                    'TO_output':                tf.TensorSpec(shape=(), dtype=tf.float32),
                }
            )
        )
        datasets.append(ds)

    full_dataset = datasets[0]
    for ds in datasets[1:]:
        full_dataset = full_dataset.concatenate(ds)

    return full_dataset.batch(BATCH_SIZE), scaler