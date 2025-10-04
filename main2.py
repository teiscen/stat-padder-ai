import os

from data_load import CSV_Data

STARTING_FOLDER         = os.path.join('.', 'Data', 'NBA', 'Roster')
PLAYERS_CSV_FILE        = os.path.join(STARTING_FOLDER, 'players.csv')
PARTICIPATED_CSV_FILE   = os.path.join(STARTING_FOLDER, 'participatedCSV.csv')
GAMES_CSV_FILE          = os.path.join(STARTING_FOLDER, 'gameCSV.csv')

def split_baskets(csv_data):
    columns_to_split = ['FG', 'FT', 'ThreePT']
    for column in columns_to_split:
        # assign the create columns to                        Split column into 2 seperate columns based on - as type int
        csv_data[[f'{column}_successes', f'{column}_attempts']] = csv_data[column].str.split('-', expand=True).astype(int)
        csv_data.drop([column], axis=1, inplace=True)

def split_awayTeamID(nba_data):
    """ awayTeamID -> oppTeamID, isHome """
    nba_data['isHome'] = (~nba_data['awayTeamID'].str.startswith('@')).astype(int)
    nba_data['awayTeamID'] = (
        nba_data['awayTeamID']
        .str.replace('@', '', regex=False)
        .str.replace('vs','', regex=False)
        .str.strip() 
        .str.lower()
    )

csv_list = [
    CSV_Data(
        PLAYERS_CSV_FILE,      
        ['playerName', 'playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src'],
    ),
    CSV_Data(
        PARTICIPATED_CSV_FILE, 
        ['FGPercent', 'ThreePTPercent', 'FTPercent'],
        split_baskets
    ),
    CSV_Data(
        GAMES_CSV_FILE,
        ['result'],
        split_awayTeamID
    )
]
merge_list = [['playerID'], ['teamID', 'gameDate']]

def add_fantasy_points(csv_data):
    csv_data['Fatasy_Pts'] = (
        csv_data['ThreePT_successes'] * 3 +
        csv_data['FG_successes'] * 2 +
        csv_data['FT_successes'] * 1 +
        csv_data['REB'] * 1.2 +
        csv_data['AST'] * 1.5 +
        csv_data['BLK'] * 2 +
        csv_data['STL'] * 2 +
        csv_data['TO'] * -1
    )

merged_pd_data = CSV_Data.merge_csv_list(csv_list, merge_list)
merged_pd_data.sort_values('gameDate')
add_fantasy_points(merged_pd_data)

from data_preprocessing import standardize_columns, embed_columns, generate_sequences

embedded_cols     = [
    'playerID', 'teamID', 'awayTeamID', 'position'
]
standardized_cols = [
    'Minutes',  'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts',
    'FT_successes', 'FT_attempts', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS'
]
feature_list = [
    'playerID', 'position', 'teamID', 'awayTeamID', 'isHome', 'Minutes',
    'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts', 
    'FT_successes', 'FT_attempts','REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS',
]

embedding_sizes = embed_columns(merged_pd_data, embedded_cols)
standardize_columns(merged_pd_data, standardized_cols)

SEQUENCE_LENGTH = 20

from model_lstm import Embedded_Model
EMBEDDED_MODEL_LIST = [
    Embedded_Model((SEQUENCE_LENGTH, ), 'int32', 'player',    embedding_sizes.item().get('playerID', 1),    50),
    Embedded_Model((SEQUENCE_LENGTH, ), 'int32', 'position',  embedding_sizes.item().get('position', 1),    10),
    Embedded_Model((SEQUENCE_LENGTH, ), 'int32', 'home_team', embedding_sizes.item().get('teamID', 1),      10),
    Embedded_Model((SEQUENCE_LENGTH, ), 'int32', 'away_team', embedding_sizes.item().get('away_teamID', 1), 10)
]




# SEQUENCES_NP_DATA, SEQUENCES_NP_LABEL = generate_sequences(MERGED_PD_DATA, FEATURE_LIST, ['playerID'], SEQUENCE_LENGTH)

# player_input         = SEQUENCES_NP_DATA[:, :, 0]
# position_input       = SEQUENCES_NP_DATA[:, :, 1]
# home_team_input      = SEQUENCES_NP_DATA[:, :, 2]
# away_team_input      = SEQUENCES_NP_DATA[:, :, 3]
# other_features_input = SEQUENCES_NP_DATA[:, :, 4:]

# X = [
#     player_input.astype('int32'),            # shape: (num_samples, SEQUENCE_LENGTH)
#     position_input.astype('int32'),          # shape: (num_samples, SEQUENCE_LENGTH)
#     home_team_input.astype('int32'),         # shape: (num_samples, SEQUENCE_LENGTH)
#     away_team_input.astype('int32'),         # shape: (num_samples, SEQUENCE_LENGTH)
#     other_features_input.astype('float32')   # shape: (num_samples, SEQUENCE_LENGTH, other_features_dim)
# ]

# y = SEQUENCES_NP_LABEL

# split_idx = int(0.8 * len(y))
# X_train = [arr[:split_idx] for arr in X]
# X_test  = [arr[split_idx:] for arr in X]
# y_train = y[:split_idx]
# y_test  = y[split_idx:]

# for feat in X_train:
#     print("\n", feat.dtype, np.any( feat == None), np.any(np.isnan(feat)))

# model = load_model('./Data/model/LSTM.keras')

# predictions = model.predict(X_test)

# for i in range(10):
#     print(f"Prediction: {predictions[i][0]:.4f} | Actual: {y_test[i]:.4f}")




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

# def get_processed_data():
#     sequences = np.load('./Data/Formatted/sequences.npy', allow_pickle=True)
#     labels    = np.load('./Data/Formatted/labels.npy', allow_pickle=True)
#     return sequences, labels
# nba_data = get_processed_data()
# vocab_sizes = np.load('./Data/Formatted/embedding_input_dims.npy', allow_pickle=True)

# Calculate vocab sizes for embeddings
# normalized_col_count = 14
# binary_col_count = 1 
# other_features_dim = normalized_col_count + binary_col_count

# vocab_sizes = {}
# embedded_columns = ['playerID', 'position', 'teamID', 'awayTeamID']
# for col in embedded_columns:
#     vocab_sizes[col] = nba_data[col].nunique()