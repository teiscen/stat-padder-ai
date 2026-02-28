import argparse

def main(argv=None):
    p = argparse.ArgumentParser(description="Process NBA CSVs into merged dataset with fantasy label.")
    p.add_argument('--output', help='Output CSV path. If empty won\'t save.', default=None)
    p.add_argument('--debug',  help='Print debug info',                       action='store_true')
    args = p.parse_args(argv)

    output = args.output
    debug  = args.debug
    
    merged_pd_data = build_data()
    add_fantasy_points(merged_pd_data)

    if output is not None:
        merged_pd_data.to_csv(output, index=False)

    if debug:
        print('Column names + first row:')
        print(merged_pd_data.columns.tolist())
        print(merged_pd_data.iloc[0])

if __name__ == "__main__":
    main()    
