import os
from util.data_load import CSV_Data

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

def get_data():
    return merged_pd_data
