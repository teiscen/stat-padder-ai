import pandas as pd
import os

"""
Data Format:
player_id, position, --injury_status--, team_id, 
Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS
"""
STARTING_FOLDER         = os.path.join('.', 'Data', 'NBA')
INJURIES_CSV_FILE       = os.path.join(STARTING_FOLDER, 'News',   'injuryReport.csv') 
PLAYERS_CSV_FILE        = os.path.join(STARTING_FOLDER, 'Roster', 'players.csv')
GAMES_CSV_FILE          = os.path.join(STARTING_FOLDER, 'Roster', 'gameCSV.csv')
PARTICIPATED_CSV_FILE   = os.path.join(STARTING_FOLDER, 'Roster', 'participatedCSV.csv')

# Load the data, and drop the columns that are unneeded 
""" 
Injury Columns : playerName, position, returnDate, status, comment
Columns To Drop:             position,             status, 
Notes:  Need to use this to constuct the game data with the appropriate status level.
            i.e. Player was injured, or fine,
        Need to parse out date from the comment to determine when the injury took place
            create accurate labeling
"""
#TODO - Need to determine how the injury labelling will work. Discuss with Paul
# injury_column_to_drop = ['position', 'status']
# injuries_data = pd.read_csv(INJURIES_CSV_FILE)
# injuries_data = injuries_data.drop(injury_column_to_drop)

"""
Players Columns: playerName, position, playerURL, teamName, teamURL, teamID, playerID, status, src
Columns to Drop: playerName            playerURL, teamName, teamURL, teamID,           status, src
"""
players_column_to_drop = ['playerName', 'playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']
players_data = pd.read_csv(PLAYERS_CSV_FILE)
players_data = players_data.drop(players_column_to_drop)

"""
Participated Columns: teamID, gameDate, Minutes, FG, FGPercent, ThreePT, ThreePTPercent, FT, FTPercent, REB, AST, BLK, STL, PF, TO, PTS, playerID
Columns to Drop     :                                FGPercent,          ThreePTPercent,     FTPercent
Notes:  Need to parse FG, ThreePT, FT, into _successes and _attempts for each
"""
participated_data_columns_to_drop = ['FGPercent', 'ThreePTPercent', 'FTPercent']
participated_data = pd.read_csv(PARTICIPATED_CSV_FILE)
participated_data = participated_data.drop(participated_data_columns_to_drop)

"""
Games Columns  : teamID, gameDate, awayTeamID, result
Columns to Drop:                               result
Notes:  Need to add a column indicating home or away
"""
games_data_columns_to_drop = ['result']
games_data = pd.read_csv(GAMES_CSV_FILE)
games_data = games_data.drop(games_data_columns_to_drop)

# Merge on the data sets
"""
How:
    'left': All rows from the left DataFrame, matching rows from the right.
    'right': All rows from the right DataFrame, matching rows from the left.
    'inner': Only rows with keys in both DataFrames.
    'outer': All rows from both DataFrames, fill missing with NaN.
"""
# teamID, gameDate, Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS, playerID, position
players_participated_data = pd.merge(players_data, participated_data, on='player_id', how='inner')
del players_data, participated_data

# teamID, gameDate, Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS, playerID, position, awayTeamID, 
merged_data = pd.merge(players_participated_data, games_data, on=['teamId', 'gameDate'], how='inner')
del players_participated_data, games_data

# To return dataset
def get_merged_data():
    return merged_data