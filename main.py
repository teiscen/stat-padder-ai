import numpy as np
# from model_lstm import model
from keras.models import load_model
# import data_preprocessing
TF_ENABLE_ONEDNN_OPTS=0

# Load and preprocess data
def get_processed_data():
    sequences = np.load('./Data/Formatted/sequences.npy', allow_pickle=True)
    labels    = np.load('./Data/Formatted/labels.npy', allow_pickle=True)
    return sequences, labels
sequences, labels = get_processed_data()
print(sequences.dtype)
print(sequences[0,0,4])  # Should print 0 or 1, not True/False
print(type(sequences[0,0,4]))  # Should be <class 'numpy.int64'> or similar, NOT bool

# Split sequences into model inputs
# Adjust indices if your feature order is different!
player_input         = sequences[:, :, 0]
position_input       = sequences[:, :, 1]
home_team_input      = sequences[:, :, 2]
away_team_input      = sequences[:, :, 3]
other_features_input = sequences[:, :, 4:]


# Prepare model inputs and labels
X = [
    player_input.astype('int32'),            # shape: (num_samples, SEQUENCE_LENGTH)
    position_input.astype('int32'),          # shape: (num_samples, SEQUENCE_LENGTH)
    home_team_input.astype('int32'),         # shape: (num_samples, SEQUENCE_LENGTH)
    away_team_input.astype('int32'),         # shape: (num_samples, SEQUENCE_LENGTH)
    other_features_input.astype('float32')   # shape: (num_samples, SEQUENCE_LENGTH, other_features_dim)
]
# y = data['labels']              # shape: (num_samples,)
y = labels

print(player_input.shape, player_input.dtype)
print(position_input.shape, position_input.dtype)
print(home_team_input.shape, home_team_input.dtype)
print(away_team_input.shape, away_team_input.dtype)
print(other_features_input.shape, other_features_input.dtype)


# Optionally, split into train/test sets
split_idx = int(0.8 * len(y))
X_train = [arr[:split_idx] for arr in X]
X_test  = [arr[split_idx:] for arr in X]
y_train = y[:split_idx]
y_test  = y[split_idx:]

print([arr.dtype for arr in X_train])
print([arr.shape for arr in X_train])
# Train the model
model = load_model('./Data/model/LSTM.keras')
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Evaluate
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")