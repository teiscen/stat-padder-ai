import argparse
from util.data_preprocessing import standardize_columns, embed_columns
from util.CSV_Data import CSV_Data
from model_config import create_model_manager
import pickle

def main(argv=None):
    p = argparse.ArgumentParser(description="Process NBA CSVs into merged dataset with fantasy label.")
    p.add_argument('--input', help='Inputh CSV paath.',                      default=None)
    p.add_argument('--output',help='Output CSV path. Won\'t save if empty.', default=None)
    p.add_argument('--debug', help='Print debug info',                       action='store_true')
    args = p.parse_args(argv)

    input = args.input
    output= args.output
    debug = args.debug

    if input is None:
        assert ValueError("Input file cannot be empty.")
    
    merged_pd_data = CSV_Data(input, []).getData()
    standardize_columns(
        merged_pd_data, 
        [
            'Minutes', 'FG_successes', 'FG_attempts', 'ThreePT_successes', 'ThreePT_attempts',
            'FT_successes', 'FT_attempts', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO', 'PTS'
        ]
    )
    embedding_sizes = embed_columns(
        merged_pd_data, 
        ['playerID', 'teamID', 'awayTeamID', 'position']
    )
    model_manager = create_model_manager(embedding_sizes)

    if output is not None:
       with open(output, 'wb') as outp:
           pickle.dump(model_manager, outp, pickle.HIGHEST_PROTOCOL)

    if debug:
        pass
        
if __name__ == "__main__":
    main()    
