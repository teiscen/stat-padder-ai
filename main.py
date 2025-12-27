




# Split features for model input
player_input         = SEQUENCES_NP_DATA[:, :, 0]
position_input       = SEQUENCES_NP_DATA[:, :, 1]
teamID_input         = SEQUENCES_NP_DATA[:, :, 2]
awayTeamID_input     = SEQUENCES_NP_DATA[:, :, 3]
isHome_input         = SEQUENCES_NP_DATA[:, :, 4]

X = [
    player_input.astype('int32'),
    position_input.astype('int32'),
    teamID_input.astype('int32'),
    awayTeamID_input.astype('int32'),
    isHome_input.astype('int32')
]
y = SEQUENCES_NP_LABEL

# Split into train/test sets
split_idx = int(0.8 * len(y))
X_train = [arr[:split_idx] for arr in X]
X_test  = [arr[split_idx:] for arr in X]
y_train = y[:split_idx]
y_test  = y[split_idx:]

