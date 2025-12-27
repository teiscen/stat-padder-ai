import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def embed_columns(csv_data, columns_to_embed):
    for col in columns_to_embed:
        csv_data[col] = csv_data[col].astype('category').cat.codes

    return {col: csv_data[col].nunique() for col in columns_to_embed}

def standardize_columns(csv_data, columns_to_standardize):
    scaler = StandardScaler()
    csv_data[columns_to_standardize] = scaler.fit_transform(csv_data[columns_to_standardize])

# Does not ensure that the data is in order
# TODO: Determine how when to swap pd and np
def generate_sequences(csv_data, sequence_length, primary_key, target_value, feature_list, label_list):
    try:
        data = csv_data[csv_data[primary_key] == target_value]
        if sequence_length > data.shape[0]:
            print(f'There is not enough data to generate a sequence for: {primary_key}')
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    sequences, labels = [], []

    for i in range(len(data) - sequence_length):
        sequences.append(data[feature_list].iloc[i:i+sequence_length].values)
        labels.append(data[label_list].iloc[i+sequence_length].values)

    return np.array(sequences), np.array(labels)


def generate_all_sequences(csv_data, sequence_length, primary_key, feature_list, label_list):
    sequences = []
    labels = []
    unique_keys = csv_data[primary_key].unique()
    for key in unique_keys:
        seq, lbl = generate_sequences(csv_data, sequence_length, primary_key, key, feature_list, label_list)
        if seq is not None and lbl is not None:
            sequences.append(seq)
            labels.append(lbl)
    if sequences and labels:
        return np.concatenate(sequences), np.concatenate(labels)
    else:
        return None, None

