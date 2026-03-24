# model_train.py
import argparse
import joblib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import keras
from config.nba_data import generate_sequences

def main(argv=None):
    p = argparse.ArgumentParser(description="Train the model and save it.")
    p.add_argument('--model-path', help='Location of saved model',        required=True)
    p.add_argument('--output',     help='Location to save trained model', required=True)
    p.add_argument('--csv-path',   help='Path to merged CSV file',        required=True)
    p.add_argument('--epochs',     help='Number of training epochs',      type=int, default=10)
    args = p.parse_args(argv)

    model   = keras.saving.load_model(args.model_path)
    dataset, scaler = generate_sequences(csv_file=args.csv_path)

    model.compile(
        optimizer='adam',
        loss={
            'Fantasy_pts_output':       'mse',
            'ThreePT_successes_output': 'mse',
            'FG_successes_output':      'mse',
            'FT_successes_output':      'mse',
            'REB_output':               'mse',
            'AST_output':               'mse',
            'BLK_output':               'mse',
            'STL_output':               'mse',
            'TO_output':                'mse',
        },
        metrics={'Fantasy_pts_output': 'mae'}
    )
    model.fit(dataset, epochs=args.epochs)
    model.save(args.output)
    print(f"Trained model saved to {args.output}")
    
    joblib.dump(scaler, args.output.replace('.keras', '_scaler.pkl'))
    print(f"Trained model saved to {args.output}")

if __name__ == "__main__":
    main()