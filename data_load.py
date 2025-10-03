import pandas as pd
import os

STARTING_FOLDER         = os.path.join('.', 'Data', 'NBA')
INJURIES_CSV_FILE       = os.path.join(STARTING_FOLDER, 'News',   'injuryReport.csv') 
PLAYERS_CSV_FILE        = os.path.join(STARTING_FOLDER, 'Roster', 'players.csv')
GAMES_CSV_FILE          = os.path.join(STARTING_FOLDER, 'Roster', 'gameCSV.csv')
PARTICIPATED_CSV_FILE   = os.path.join(STARTING_FOLDER, 'Roster', 'participatedCSV.csv')
DEBUG_PRINT = False

# Load Data
def get_players_data():
    """ Players Columns: position, playerID """
    players_column_to_drop = ['playerName', 'playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']
    players_data = pd.read_csv(PLAYERS_CSV_FILE)
    players_data = players_data.drop(players_column_to_drop, axis=1)#, errors='ignore')
    
    if(DEBUG_PRINT): print("Players Columns:", players_data.columns.tolist())
    return players_data

def get_participated_data():
    """ Participated Columns: teamID, gameDate, Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS, playerID """
    participated_data_columns_to_drop = ['FGPercent', 'ThreePTPercent', 'FTPercent']
    participated_data = pd.read_csv(PARTICIPATED_CSV_FILE)
    participated_data = participated_data.drop(participated_data_columns_to_drop, axis=1)#, errors='ignore' )
    
    if(DEBUG_PRINT): print("Participated columns:", participated_data.columns.tolist())
    return participated_data

def get_games_data():
    """ Games Columns: teamID, gameDate, awayTeamID """
    games_data_columns_to_drop = ['result']
    games_data = pd.read_csv(GAMES_CSV_FILE)
    games_data = games_data.drop(games_data_columns_to_drop, axis=1)#, errors='ignore')
    
    if(DEBUG_PRINT): print("Games Columns:", games_data.columns.tolist())
    return games_data

# Merge Data
def get_merged_data():
    """ 
    Merged Columns: 
    playerID, position, gameDate, teamID, awayTeamID, 
    Minutes, FG, ThreePT, FT, REB, AST, BLK, STL, PF, TO, PTS,  
    """
    player_data = get_players_data()
    participated_data = get_participated_data()
    games_data = get_games_data()
    merged_data = pd.merge(
                    pd.merge(player_data, participated_data, on='playerID', how='inner'), 
                    games_data, on=['teamID', 'gameDate'], how='inner')

    del player_data, participated_data, games_data
    if(DEBUG_PRINT): print("Merged Columns: ", merged_data.columns.tolist())
    return merged_data
    
# Split the data
def split_baskets(nba_data):
    """ FG, FT, ThreePT -> X_successes, X_attempts """
    columns_to_split = ['FG', 'FT', 'ThreePT']
    for column in columns_to_split:
        # assign the create columns to                          Split column into 2 seperate columns based on - as type int
        nba_data[[f'{column}_successes', f'{column}_attempts']] = nba_data[column].str.split('-', expand=True).astype(int)
        nba_data.drop([column], axis=1, inplace=True)

    if(DEBUG_PRINT): print("Split Baskets Columns: ", nba_data.columns.tolist())

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

    if(DEBUG_PRINT): print("Split AwayTeam Columns: ", nba_data.columns.tolist())

# Add a score
# TODO: Find a better way of handling formulas
def add_label(nba_data, formula=None):
    if formula is None:
        formula = lambda ThreePT, FG, FT, REB, AST, BLK, STL, TO: (
            (ThreePT * 3) + (FG * 2) + (FT * 1) + (REB * 1.2) + (AST * 1.5) + (BLK * 2) + (STL * 2) + (TO * -1)
        )

    """ Formula implements default point scoring. Can be customized. """
    nba_data['fantasy_points'] = nba_data.apply(
        lambda row: formula(row['ThreePT_successes'], row['FG_successes'], row['FT_successes'], 
                            row['REB'], row['AST'], row['BLK'], row['STL'], row['TO']),
        axis=1
    )

def get_nba_data(formula=None):
    formula = None # TODO: Get rid of when a solution for labelling is found
    
    nba_data = get_merged_data()
    split_baskets(nba_data)
    split_awayTeamID(nba_data)
    add_label(nba_data, formula)  # Uses default formula
    return nba_data