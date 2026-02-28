from util.data_preprocessing import standardize_columns, embed_columns, generate_all_sequences

from data import get_data
merged_pd_data = get_data()


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
label_list = [
    
]

embedding_sizes = embed_columns(merged_pd_data, embedded_cols)
standardize_columns(merged_pd_data, standardized_cols)

def get_embed():
    return embedding_sizes