import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_embedded_columns(nba_data):
    """ returns the input dims of the embedded columns"""
    embedded_columns = ['playerID', 'teamID', 'awayTeamID', 'position']
    for col in embedded_columns:
        nba_data[col] = nba_data[col].astype('category').cat.codes

    return {col: nba_data[col].nunique() for col in embedded_columns}

# Standardizing
def create_standardized_columns(nba_data):
    standardized_columns = [
        'Minutes', 'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts',
        'FT_successes', 'FT_attempts', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS'
    ]
    scaler = StandardScaler()
    nba_data[standardized_columns] = scaler.fit_transform(nba_data[standardized_columns])


# Create the Sequences
def generate_sequences(nba_data, sequence_length=20): 
    # Ignore players with less than the sequence length
    nba_data = nba_data.groupby('playerID').filter(lambda g: len(g) > sequence_length + 1) 

    featuresList = [
        'playerID', 'position', 'teamID', 'awayTeamID', 'isHome', 'Minutes',
        'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts', 
        'FT_successes', 'FT_attempts','REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS',
    ]
    sequences    = []
    labels       = []

    for playerID, group in nba_data.groupby('playerID'):
        group = group.sort_values('gameDate')
        for i in range(len(group) - sequence_length):
            seq = group[featuresList].iloc[i:i+sequence_length].values
            label = group['fantasy_points'].iloc[i+sequence_length]
            sequences.append(seq)
            labels.append(label)

    return sequences, labels

# # Convert to arrays
# sequences_array = np.array(sequences)
# labels_array = np.array(labels)

# sequences_array[:, :, 0] = sequences_array[:, :, 0].astype('int32')  # playerID
# sequences_array[:, :, 1] = sequences_array[:, :, 1].astype('int32')  # position
# sequences_array[:, :, 2] = sequences_array[:, :, 2].astype('int32')  # teamID
# sequences_array[:, :, 3] = sequences_array[:, :, 3].astype('int32')  # awayTeamID

# print("Example sequence:\n", sequences_array[0])

# print("isHome dtype in sequences_array:", sequences_array[:, :, 4].dtype)
# print("Unique values in isHome:", np.unique(sequences_array[:, :, 4]))

# # Save to disk
# np.save('Data/Formatted/sequences.npy', sequences_array)
# np.save('Data/Formatted/labels.npy', labels_array)
# np.save('Data/Formatted/embedding_input_dims.npy', embedding_input_dims)
