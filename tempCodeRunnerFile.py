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
print(players_data.columns.tolist())