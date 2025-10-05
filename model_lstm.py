from keras import layers
from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Masking
from keras.models import Model
# import data_preprocessing
import numpy as np

dense_output = Dense(32, activation='relu', name='hidden_dense')(lstm_output)
final_output = Dense(1, name='output_layer')(dense_output)

model = Layer( 
    inputs=[player_input, position_input, home_team_input, away_team_input, other_features_input],
    outputs=final_output,
    name='nba_lstm_with_embeddings'
)
model.compile(optimizer='adam', loss='mean_squared_error')
model.save('./Data/model/LSTM.keras')  


# Gist of it:
# Input layer has all the stats that it will be trained on, 
# these are fed into Embedding and isHome as those are what we know, 
# Concatenate(axis=-1, name='')([Layers]]) merges them together at that point
# The dense and final output can be changed and expiremented with. 



#TODO:Find a better name 
class Layer_Factory:
    def __init__(): pass

    def create_input(shape, dataType, name):
        return Input(shape=shape, dtype=dataType, name=name)
   
    def create_embedded(name, layer, sequence_len, input_dim, output_dim, isMasking=False):
        return Embedding(
            name = name,
            input_length = sequence_len,
            input_dim = input_dim,
            output_dim = output_dim,
            mask_zero = isMasking
        )(layer)

    def create_lstm(name, layer, units):
        # Required for GPU accel
        activation = 'tanh', rec_activation = 'sigmoid'
        unroll = False, use_bias = True
        rec_dropout = 0
        return LSTM(
                units,                    
                activation=activation,   
                recurrent_activation=rec_activation,
                recurrent_dropout=rec_dropout,      
                unroll=unroll,       
                use_bias=use_bias,   
                name=name,
        )(layer)        

    def concatenate_layers(name, layer_list):
        return Concatenate(axis=-1, name=name)([layer_list])

    def create_model(name, input_layer_list, output_layer_list, )
