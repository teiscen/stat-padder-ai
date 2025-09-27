import pandas as pd
from sklearn.preprocessing import StandardScaler
import data_load
import numpy as np


""" teamID, gameDate, Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS, playerID, position, awayTeamID, """
nba_data = data_load.get_merged_data()
# print("NBA Data Colums: ", nba_data.columns.tolist())

""" ..., FG_successes, GF_attempts, ThreePT_successes, ThreePT_attempts, FT_successes, FT_attempts, ... """
# split FG, ThreePT, FT -> _successes and _attempts
columns_to_split = ['FG', 'FT', 'ThreePT']
for column in columns_to_split:
    # assign the create columns to                          Split column into 2 seperate columns based on - as type int
    nba_data[[f'{column}_successes', f'{column}_attempts']] = nba_data[column].str.split('-', expand=True).astype(int)
    nba_data.drop([column], axis=1, inplace=True)

""" ..., awayTeamID, isHome """
# split awayTeamID -> oppTeamID and isHome 
nba_data['isHome'] = (~nba_data['awayTeamID'].str.startswith('@')).astype(int)
nba_data['awayTeamID'] = (
    nba_data['awayTeamID']
    .str.replace('@', '', regex=False)
    .str.replace('vs','', regex=False)
    .str.strip() 
    .str.lower()
)

# Encode the features ... 
"""
    playerID,          int,     embedding     
    position,          int,     embedding   
    gameDate,          ---,     Drop     
    teamID,            int,     embedding
    awayTeamID,        int,     embedding         
    isHome             bool,    Binary 
    Minutes,           int,     Normalize    
    FG_successes,      int,     Normalize        
    FG_attempts,       int,     Normalize        
    ThreePT_successes, int,     Normalize            
    ThreePT_attempts,  int,     Normalize            
    FT_successes,      int,     Normalize        
    FT_attempts        int,     Normalize        
    REB,               int,     Normalize
    AST,               int,     Normalize
    BLK,               int,     Normalize
    STL,               int,     Normalize
    PF,                int,     Normalize
    TO,                int,     Normalize
    PTS,               int,     Normalize
"""
# One-Hot Encoding
# one_hot_columns = ['position']
# nba_data = pd.get_dummies(nba_data, columns=one_hot_columns)

# Embeddings
embedded_columns = ['playerID', 'teamID', 'awayTeamID', 'position']
for col in embedded_columns:
    nba_data[col] = nba_data[col].astype('category').cat.codes
embedding_input_dims = {col: nba_data[col].nunique() for col in embedded_columns}

# Standardizing
standardized_columns = [
    'Minutes', 'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts',
    'FT_successes', 'FT_attempts', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS'
]
scaler = StandardScaler()
nba_data[standardized_columns] = scaler.fit_transform(nba_data[standardized_columns])

# Labeling
def formula(ThreePT, FG, FT, REB, AST, BLK, STL, TO):
    return (ThreePT*3) + (FG*2) + (FT*1) + (REB*1.2) + (AST*1.5) + (BLK*2) + (STL*2) + (TO*-1)

nba_data['fantasy_points'] = nba_data.apply(
    lambda row: formula( row['ThreePT_successes'], row['FG_successes'], row['FT_successes'], 
                        row['REB'], row['AST'], row['BLK'], row['STL'], row['TO']),
    axis=1
)

# Create the Sequences
"""
playerID, position,
    [... game 1 ...], 
    ...,
    [... game SEQUENCE_LENGTH ...]
"""
SEQUENCE_LENGTH = 20
nba_data = nba_data.groupby('playerID').filter(lambda g: len(g) > SEQUENCE_LENGTH + 1) # Ignore players with less than the sequence length

sequences = []
labels = []

featuresList = [
    'playerID', 'position',
    'teamID', 'awayTeamID', 'isHome', 'Minutes',
    'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts', 'FT_successes', 'FT_attempts',
    'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS',
]

for playerID, group in nba_data.groupby('playerID'):
    group = group.sort_values('gameDate')
    for i in range(len(group) - SEQUENCE_LENGTH):
        seq = group[featuresList].iloc[i:i+SEQUENCE_LENGTH].values
        label = group['fantasy_points'].iloc[i+SEQUENCE_LENGTH]
        sequences.append(seq)
        labels.append(label)

# testing
# print(f"Number of sequences: {len(sequences)}")
# print(f"Shape of one sequence: {sequences[0].shape}")  # Should be (SEQUENCE_LENGTH, number_of_features)
# print(f"Type of sequences: {type(sequences)}")

# print("First sequence:\n", np.array(sequences[0]))
# print("First label:", labels[0])

# Added code block
# player_games = nba_data.groupby('playerID').filter(lambda g: len(g) > SEQUENCE_LENGTH)
# print("Expected label:", player_games['fantasy_points'].iloc[SEQUENCE_LENGTH])
# print("Actual label:", labels[0])

# Convert to arrays
sequences_array = np.array(sequences)
labels_array = np.array(labels)

sequences_array[:, :, 0] = sequences_array[:, :, 0].astype('int32')  # playerID
sequences_array[:, :, 1] = sequences_array[:, :, 1].astype('int32')  # position
sequences_array[:, :, 2] = sequences_array[:, :, 2].astype('int32')  # teamID
sequences_array[:, :, 3] = sequences_array[:, :, 3].astype('int32')  # awayTeamID

print("Example sequence:\n", sequences_array[0])

print("isHome dtype in sequences_array:", sequences_array[:, :, 4].dtype)
print("Unique values in isHome:", np.unique(sequences_array[:, :, 4]))

# Save to disk
np.save('Data/Formatted/sequences.npy', sequences_array)
np.save('Data/Formatted/labels.npy', labels_array)
np.save('Data/Formatted/embedding_input_dims.npy', embedding_input_dims)
