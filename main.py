import numpy as np
from model_lstm import model
import data_preprocessing

# Load and preprocess data
data = data_preprocessing.get_processed_data()

# Prepare model inputs and labels
X = [
    data['player_input'],        # shape: (num_samples, SEQUENCE_LENGTH)
    data['position_input'],      # shape: (num_samples, SEQUENCE_LENGTH)
    data['home_team_input'],     # shape: (num_samples, SEQUENCE_LENGTH)
    data['away_team_input'],     # shape: (num_samples, SEQUENCE_LENGTH)
    data['other_features_input'] # shape: (num_samples, SEQUENCE_LENGTH, other_features_dim)
]
y = data['labels']              # shape: (num_samples,)

# Optionally, split into train/test sets
from sklearn.model_selection import train_test_split
X_train = [arr[:int(0.8*len(arr))] for arr in X]
X_test = [arr[int(0.8*len(arr)):] for arr in X]
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(y)):]

# Train the model
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