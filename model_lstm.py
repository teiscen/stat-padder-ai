from keras import layers
from keras.layers import LSTM 

import data_preprocessing

"""
    playerID,          Embedding     
    position,          One-hot encoding     
    gameDate,          Drop     
    teamID,            Embedding
    awayTeamID,        Embedding         
    isHome             Binary 
    Minutes,           Normalize    
    FG_successes,      Normalize        
    FG_attempts,       Normalize        
    ThreePT_successes, Normalize            
    ThreePT_attempts,  Normalize            
    FT_successes,      Normalize        
    FT_attempts        Normalize        
    REB,               Normalize
    AST,               Normalize
    BLK,               Normalize
    STL,               Normalize
    PF,                Normalize
    TO,                Normalize
    PTS,               Normalize
"""
nba_data = data_preprocessing.get_processed_data()


