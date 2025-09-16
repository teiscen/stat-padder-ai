import pandas as pd
from sklearn.preprocessing import StandardScaler
import data_load

""" teamID, gameDate, Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS, playerID, position, awayTeamID, """
nba_data = data_load.get_merged_data()

""" ..., FG_successes, GF_attempts, ThreePT_successes, ThreePT_attempts, FT_successes, FT_attempts, ... """
# split FG, ThreePT, FT -> _successes and _attempts
columns_to_split = ['FG', 'FT', 'ThreePT']
for column in columns_to_split:
    # assign the create columns to                          Split column into 2 seperate columns based on - as type int
    nba_data[[f'{column}_successes', f'{column}_attempts']] = nba_data[column].str.split('-', expand=True).astype(int)
    nba_data.drop([column])

""" ..., awayTeamID, isHome """
# split awayTeamID -> oppTeamID and isHome 
nba_data['isHome'] = ~nba_data['awayTeamID'].str.startswith('@')
nba_data['awayTeamID'] = (
    nba_data['awayTeamID']
    .str.replace('@', regex=False)
    .str.replace('vs', regex=False)
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
    nba_data[col] = nba_data[col].asType('category').cat.codes
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

# nba_data['fantasy_points'] = nba_data.apply(
#     lambda row: formula( row['ThreePT_successes'], row['FG_successes'], row['FT_successes'], 
#                         row['REB'], row['AST'], row['BLK'], row['STL'], row['TO']),
#     axis=1
# )

def get_processed_data():
    return nba_data