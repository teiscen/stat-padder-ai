# model_build.py
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nba_model import build_model

def main(argv=None):
    p = argparse.ArgumentParser(description="Build the model and save it.")
    p.add_argument('--input',      help='CSV file of the merged data.',       required=True)
    p.add_argument('--output',     help='Location where model will be saved', required=True)
    p.add_argument('--is-masking', help='Use masking', type=lambda x: x.lower() == 'true', default=False)    
    args = p.parse_args(argv)

    model = build_model(csv_file=args.input, isMasking=args.is_masking)
    model.save(args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()