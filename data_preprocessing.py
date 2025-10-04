import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def embed_columns(csv_data, columns_to_embed):
    for col in columns_to_embed:
        csv_data[col] = csv_data[col].astype('category').cat.codes

    return {col: csv_data[col].nunique() for col in columns_to_embed}

# Standardizing
def standardize_columns(csv_data, columns_to_standardize):
    scaler = StandardScaler()
    csv_data[columns_to_standardize] = scaler.fit_transform(csv_data[columns_to_standardize])

# TODO: Change so that it doesnt generate it all at once.
# Create the Sequences
def generate_sequences(csv_data, featureList, primaryFilter, sequence_length=20): 
    # Ignore players with less than the sequence length
    csv_data = csv_data.groupby(primaryFilter).filter(lambda g: len(g) > sequence_length + 1) 

    sequences, labels = [], []

    for _, group in csv_data.groupby(primaryFilter):
        for i in range(len(group) - sequence_length):
            sequences.append(group[featureList].iloc[i:i+sequence_length].values)
            labels.append(group['Label'].iloc[i+sequence_length])

    return np.array(sequences), np.array(labels)


