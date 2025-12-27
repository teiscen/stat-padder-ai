import numpy as np
# from model_lstm import model
from keras.models import load_model
# import data_preprocessing

# Load and preprocess data
def get_processed_data():
    sequences = np.load('./Data/Formatted/sequences.npy', allow_pickle=True)
    labels    = np.load('./Data/Formatted/labels.npy', allow_pickle=True)
    return sequences, labels
sequences, labels = get_processed_data()
#print(sequences.dtype)
# print(sequences[0,0,4])  # Should print 0 or 1, not True/False
# print(type(sequences[0,0,4]))  # Should be <class 'numpy.int64'> or similar, NOT bool

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
# for feature in X:
    # print(feature.dtype, feature.shape)

# Optionally, split into train/test sets
split_idx = int(0.8 * len(y))
X_train = [arr[:split_idx] for arr in X]
X_test  = [arr[split_idx:] for arr in X]
y_train = y[:split_idx]
y_test  = y[split_idx:]



for feat in X_train:
    print("\n", feat.dtype, np.any( feat == None), np.any(np.isnan(feat)))

# print([arr.dtype for arr in X_train])
# print([arr.shape for arr in X_train])
# Train the model
model = load_model('./Data/model/LSTM.keras')
# model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_test, y_test),
#     epochs=10,
#     batch_size=32
# )
# model.save('./Data/model/LSTM.keras')

# X_test is your test input data (same format as training)
predictions = model.predict(X_test)

# predictions will be a NumPy array of shape (num_samples, 1)

# Print predictions and actual values side by side for first 10 samples
for i in range(10):
    print(f"Prediction: {predictions[i][0]:.4f} | Actual: {y_test[i]:.4f}")

# # Evaluate
# loss = model.evaluate(X_test, y_test)
# print(f"Test loss: {loss}")