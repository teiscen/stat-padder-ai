import argparse
from util.data_preprocessing import generate_all_sequences


def main(argv=None):
    p = argparse.ArgumentParser(description="Process NBA CSVs into merged dataset with fantasy label.")
    p.add_argument('--sequence-length', default=20)
    p.add_argument('--output', help='Output CSV path. If empty won\'t save.', default=None)
    p.add_argument('--debug',  help='Print debug info',                       action='store_true')
    args = p.parse_args(argv)

    output = args.output
    debug  = args.debug
    sequence_length = args.sequence_length



    SEQUENCE_NP_DATA, SEQUENCES_NP_LABEL = generate_all_sequences(
        , 
        sequence_length, 
        'playerID', 

    )


    # if output is not None:
    # if debug:

if __name__ == "__main__":
    main()    
