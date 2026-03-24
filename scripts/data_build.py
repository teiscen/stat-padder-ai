# data_build.py
import argparse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nba_data import build_data

def main(argv=None):
    p = argparse.ArgumentParser(description="Build and save the merged NBA dataset to a CSV file.")
    p.add_argument('--output', help='Location where the CSV will be saved', required=True)
    args = p.parse_args(argv)

    data = build_data()
    data.to_csv(args.output, index=False)
    print(f"Data saved to {args.output} with shape {data.shape}")

if __name__ == "__main__":
    main()