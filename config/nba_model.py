import keras
from util.data_preprocessing import embed_columns
from util.CSV_Data import CSV_Data

SEQUENCE_LENGTH = 10

def build_model(csv_file, isMasking=True):
    embedding_sizes = embed_columns(
        CSV_Data(csv_file, []).getData(),
        ['playerID', 'awayTeamID', 'teamID', 'position']
    )

    # LSTM stream embeddings (masking enabled)
    player_embedding = keras.layers.Embedding(name='embed_player', mask_zero=isMasking, output_dim=50, input_dim=int(embedding_sizes.get('playerID')))
    posn_embedding   = keras.layers.Embedding(name='embed_posn',   mask_zero=isMasking, output_dim=50, input_dim=int(embedding_sizes.get('position')))
    team_embedding   = keras.layers.Embedding(name='embed_team',   mask_zero=isMasking, output_dim=50, input_dim=int(embedding_sizes.get('teamID')))
    opp_embedding    = keras.layers.Embedding(name='embed_opp',    mask_zero=isMasking, output_dim=50, input_dim=int(embedding_sizes.get('awayTeamID')))

    # Context stream embeddings (masking disabled - single values, not sequences)
    ctx_player_embedding = keras.layers.Embedding(name='embed_ctx_player', mask_zero=False, output_dim=50, input_dim=int(embedding_sizes.get('playerID')))
    ctx_opp_embedding    = keras.layers.Embedding(name='embed_ctx_opp',    mask_zero=False, output_dim=50, input_dim=int(embedding_sizes.get('awayTeamID')))

    # --- LSTM Stream ---
    input_player = keras.Input(name='input_player', dtype='int32',   shape=(SEQUENCE_LENGTH,))
    input_posn   = keras.Input(name='input_posn',   dtype='int32',   shape=(SEQUENCE_LENGTH,))
    input_team   = keras.Input(name='input_team',   dtype='int32',   shape=(SEQUENCE_LENGTH,))
    input_opp    = keras.Input(name='input_opp',    dtype='int32',   shape=(SEQUENCE_LENGTH,))
    input_stats  = keras.Input(name='input_stats',  dtype='float32', shape=(SEQUENCE_LENGTH, 14))

    embed_player_seq = player_embedding(input_player)
    embed_posn_seq   = posn_embedding(input_posn)
    embed_team_seq   = team_embedding(input_team)
    embed_opp_seq    = opp_embedding(input_opp)

    lstm_input = keras.layers.concatenate([embed_player_seq, embed_posn_seq, embed_team_seq, embed_opp_seq, input_stats])
    lstm_out   = keras.layers.LSTM(name='lstm', units=64, activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True, recurrent_dropout=0)(lstm_input)

    # --- Context Stream ---
    input_ctx_player = keras.Input(name='input_ctx_player', dtype='int32',   shape=(1,))
    input_ctx_opp    = keras.Input(name='input_ctx_opp',    dtype='int32',   shape=(1,))
    input_isHome     = keras.Input(name='input_isHome',     dtype='float32', shape=(1,))

    embed_ctx_player = keras.layers.Flatten()(ctx_player_embedding(input_ctx_player))
    embed_ctx_opp    = keras.layers.Flatten()(ctx_opp_embedding(input_ctx_opp))

    context_concat = keras.layers.concatenate([embed_ctx_player, embed_ctx_opp, input_isHome])
    context_out    = keras.layers.Dense(name='context_dense', units=32, activation='relu')(context_concat)

    # --- Merge Streams ---
    merged       = keras.layers.concatenate([lstm_out, context_out])
    hidden_layer = keras.layers.Dense(name='hidden', units=32, activation='relu')(merged)

    # --- Output Layers ---
    Fantasy_pts_output       = keras.layers.Dense(name='Fantasy_pts_output',       units=1, activation='linear')(hidden_layer)
    ThreePT_successes_output = keras.layers.Dense(name='ThreePT_successes_output', units=1, activation='linear')(hidden_layer)
    FG_successes_output      = keras.layers.Dense(name='FG_successes_output',      units=1, activation='linear')(hidden_layer)
    FT_successes_output      = keras.layers.Dense(name='FT_successes_output',      units=1, activation='linear')(hidden_layer)
    REB_output               = keras.layers.Dense(name='REB_output',               units=1, activation='linear')(hidden_layer)
    AST_output               = keras.layers.Dense(name='AST_output',               units=1, activation='linear')(hidden_layer)
    BLK_output               = keras.layers.Dense(name='BLK_output',               units=1, activation='linear')(hidden_layer)
    STL_output               = keras.layers.Dense(name='STL_output',               units=1, activation='linear')(hidden_layer)
    TO_output                = keras.layers.Dense(name='TO_output',                units=1, activation='linear')(hidden_layer)

    return keras.Model(
        inputs=[input_player, input_posn, input_team, input_opp, input_stats,
                input_ctx_player, input_ctx_opp, input_isHome],
        outputs=[Fantasy_pts_output, ThreePT_successes_output, FG_successes_output,
                 FT_successes_output, REB_output, AST_output, BLK_output, STL_output, TO_output]
    )