from keras import layers
from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Masking
from keras.models import Model
# import data_preprocessing
import numpy as np

"""
    playerID,          Embedding     
    position,          Embedding   
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
def get_processed_data():
    sequences = np.load('./Data/Formatted/sequences.npy', allow_pickle=True)
    labels    = np.load('./Data/Formatted/labels.npy', allow_pickle=True)
    return sequences, labels
nba_data = get_processed_data()
vocab_sizes = np.load('./Data/Formatted/embedding_input_dims.npy', allow_pickle=True)

# Calculate vocab sizes for embeddings
normalized_col_count = 14
binary_col_count = 1 
other_features_dim = normalized_col_count + binary_col_count

# vocab_sizes = {}
# embedded_columns = ['playerID', 'position', 'teamID', 'awayTeamID']
# for col in embedded_columns:
#     vocab_sizes[col] = nba_data[col].nunique()

# Create Seperate Inputs
SEQUENCE_LENGTH = 20
player_input         = Input(shape=(SEQUENCE_LENGTH,), dtype='int32', name='player_input')    # Embeddings 
position_input       = Input(shape=(SEQUENCE_LENGTH,), dtype='int32', name='position_input')  # Embeddings 
home_team_input      = Input(shape=(SEQUENCE_LENGTH,), dtype='int32', name='home_team_input') # Embeddings     
away_team_input      = Input(shape=(SEQUENCE_LENGTH,), dtype='int32', name='away_team_input') # Embeddings     
other_features_input = Input(shape=(SEQUENCE_LENGTH, other_features_dim), dtype='float32', name='other_features') # Normalized + Binary

# Ouput dims is what we chose
isMasking = True
player_embedding = Embedding(
    input_dim=int(vocab_sizes.item().get('playerID', 1)),   
    output_dim=50,
    input_length=SEQUENCE_LENGTH,
    mask_zero=isMasking, 
    name='player_embedding'
)(player_input)

position_embedding = Embedding(
    input_dim=int(vocab_sizes.item().get('position', 1)),  
    output_dim=10,                      
    input_length=SEQUENCE_LENGTH,
    mask_zero=isMasking, 
    name='position_embedding'
)(position_input)

team_embedding = Embedding(
    input_dim=int(vocab_sizes.item().get('teamID', 1)),    
    output_dim=10,                      
    input_length=SEQUENCE_LENGTH,
    mask_zero=isMasking, 
    name='home_team_embedding'
)(home_team_input)

away_team_embedding = Embedding(
    input_dim=int(vocab_sizes.item().get('awayTeamID', 1)),
    output_dim=10,                      
    input_length=SEQUENCE_LENGTH,
    mask_zero=isMasking, 
    name='away_team_embedding'
)(away_team_input)

other_features_masked = Masking(mask_value=0.0, name='other_features_masking')(other_features_input)

all_features = Concatenate(axis=-1, name='feature_concat')([
    player_embedding,      # Shape: (batch, sequence, 50)
    position_embedding,    # Shape: (batch, sequence, 10)
    team_embedding,        # Shape: (batch, sequence, 10) 
    away_team_embedding,   # Shape: (batch, sequence, 10)
    other_features_masked  # Shape: (batch, sequence, other_dims)
])

# Required for GPU acceleration
LSTM_ACTIVIATION = 'tanh'
LSTM_RECURRENT_ACTIVATION = 'sigmoid'
LSTM_RECURRENT_DROPOUT = 0
LSTM_UNROLL = False
LSTM_USE_BIAS = True
#
LSTM_UNITS = 64

lstm_output = LSTM(
    LSTM_UNITS,                    
    activation=LSTM_ACTIVIATION,   
    recurrent_activation=LSTM_RECURRENT_ACTIVATION,
    recurrent_dropout=LSTM_RECURRENT_DROPOUT,      
    unroll=LSTM_UNROLL,       
    use_bias=LSTM_USE_BIAS,   
    name='lstm_layer'
)(all_features)

dense_output = Dense(32, activation='relu', name='hidden_dense')(lstm_output)
final_output = Dense(1, name='output_layer')(dense_output)

model = Model(
    inputs=[player_input, position_input, home_team_input, away_team_input, other_features_input],
    outputs=final_output,
    name='nba_lstm_with_embeddings'
)

model.compile(optimizer='adam', loss='mean_squared_error')

model.save('./Data/model/LSTM.keras')  