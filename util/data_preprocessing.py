import pandas as pd
import numpy as np

def embed_columns(csv_data, columns_to_embed):
    for col in columns_to_embed:
        csv_data[col] = csv_data[col].astype('category').cat.codes
        
    return {col: csv_data[col].nunique() for col in columns_to_embed}

