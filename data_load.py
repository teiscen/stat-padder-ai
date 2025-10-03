import pandas as pd
import os

STARTING_FOLDER         = os.path.join('.', 'Data', 'NBA')
INJURIES_CSV_FILE       = os.path.join(STARTING_FOLDER, 'News',   'injuryReport.csv') 
PLAYERS_CSV_FILE        = os.path.join(STARTING_FOLDER, 'Roster', 'players.csv')
GAMES_CSV_FILE          = os.path.join(STARTING_FOLDER, 'Roster', 'gameCSV.csv')
PARTICIPATED_CSV_FILE   = os.path.join(STARTING_FOLDER, 'Roster', 'participatedCSV.csv')
DEBUG_PRINT = False

class CSV_Data:
    def __init__(self, filePath, colDropList):
        self.filePath = filePath        
        self.colDropList = colDropList
        self.csv_data  = None
    
    def __del__(self):
        self.delData()

    def readFile(self):
        try:
            self.csv_data = pd.read_csv(self.filePath)
            self.csv_data = self.csv_data.drop(self.colDropList, axis=1)
        except FileNotFoundError:
            print(f"Error: File not found - {self.filePath}")
            self.csv_data = None
        except pd.errors.EmptyDataError:
            print(f"Error: No data - {self.filePath}")
            self.csv_data = None
        except KeyError as e:
            print(f"Error: Column(s) not found when dropping: {e}")
            self.csv_data = None
        except Exception as e:
            print(f"Unexpected error reading {self.filePath}: {e}")
            self.csv_data = None

    def getData(self):
        if self.csv_data is None:
            print(f"Warning: No data loaded from {self.filePath}")
        return self.csv_data

    def delData(self):
        try:
            del self.csv_data
            self.csv_data = None
        except AttributeError:
            print("Error: csv_data attribute does not exist.")
        except Exception as e:
            print(f"Unexpected error deleting csv_data: {e}")

    def printDebug(self):
        # baseName -> get the last part; splitext-> name.txt into [name, txt] 
        name = os.path.splitext(os.path.basename(self.filePath))[0]
        print(f"Debug: {name}, Columns:\n{self.csv_data.columns.tolist()}")

    @staticmethod
    def merge_csv_list(csv_list, merge_list):
        merge_list = [None] + merge_list
        if len(csv_list) != len(merge_list):
            raise ValueError("Length of csv_list must be one less than length of merge_list.")

        merged_data = None
        # Skip the first CSV in csv_list; start from index 1
        for csv, merge in csv_list, merge_list:
            if csv.getData() is None: csv.readFile()
            if merged_data is None:
                merged_data = csv.readFile()
                continue
            merged_data = pd.merge(merge_data, csv.getData(), on=merge, how='inner')

        return merged_data



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
# Supply the function to merge, and what to merge it on 
def merge_data(toMergeFuncList):
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
def split_baskets(nba_data)
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