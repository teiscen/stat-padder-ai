# model_predict.py
import argparse
import keras
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.nba_data import SEQUENCE_LENGTH, FEATURE_COLS, _encode_categoricals, _normalize_features

def main(argv=None):
    p = argparse.ArgumentParser(description="Predict stats for a player's next game.")
    p.add_argument('--model-path',   help='Location of trained model',      required=True)
    p.add_argument('--input',        help='CSV file with recent player data', required=True)
    p.add_argument('--player-name',  help='Player name to predict for',     required=True)
    p.add_argument('--opp-name',     help='Opponent team name (e.g. bos)',  required=True)
    p.add_argument('--is-home',      help='Is the player at home',          type=int, choices=[0,1], required=True)
    args = p.parse_args(argv)

    model = keras.saving.load_model(args.model_path)

    from config.nba_data import build_data
    csv_data = build_data()

    # Look up player name before encoding (encoding destroys string labels)
    # players.csv still has playerName so we need to find the playerID that matches
    from util.CSV_Data import CSV_Data
    from config.nba_data import PLAYERS_CSV_FILE
    players_df = CSV_Data(PLAYERS_CSV_FILE, ['playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']).getData()
    match = players_df[players_df['playerName'].str.lower() == args.player_name.lower()]
    if match.empty:
        print(f"Player '{args.player_name}' not found.")
        return
    raw_player_id = match.iloc[0]['playerID']

    # Find the encoded player ID by checking what LabelEncoder assigned
    # Since build_data already encoded, find the row in csv_data matching raw playerID
    # We need the pre-encoded csv to map name -> encoded id, so reload and encode
    from config.nba_data import STARTING_FOLDER, GAMES_CSV_FILE, PARTICIPATED_CSV_FILE
    from config.nba_data import _split_baskets, _split_awayTeamID, _add_fantasy_points
    from sklearn.preprocessing import LabelEncoder

    # Re-encode to get the same mapping as training
    _encode_categoricals(csv_data)
    _normalize_features(csv_data)

    # Find encoded player ID by matching raw playerID before encoding
    # Instead, re-read raw and match
    raw_players = CSV_Data(PLAYERS_CSV_FILE, ['playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']).getData()
    raw_ids = sorted(csv_data['playerID'].unique())  # encoded IDs are 0-based sorted

    # Map awayTeamID name to encoded ID
    opp_name_clean = args.opp_name.strip().lower()
    opp_rows = csv_data[csv_data['awayTeamID'] == opp_name_clean] if 'awayTeamID' in csv_data.columns else None

    # Simpler approach: rebuild csv_data without encoding to get the mapping
    from config.nba_data import build_data as _build_raw
    import pandas as pd
    csv_list_raw = [
        CSV_Data(PLAYERS_CSV_FILE,      ['playerName', 'playerURL', 'teamName', 'teamURL', 'teamID', 'status', 'src']),
        CSV_Data(PARTICIPATED_CSV_FILE, ['FGPercent', 'ThreePTPercent', 'FTPercent'], _split_baskets),
        CSV_Data(GAMES_CSV_FILE,        ['result'], _split_awayTeamID)
    ]
    from util.CSV_Data import CSV_Data as CSD
    raw_merged = CSD.merge_csv_list(csv_list_raw, [['playerID'], ['teamID', 'gameDate']])
    raw_merged = raw_merged.sort_values(["playerID", "gameDate"])
    _add_fantasy_points(raw_merged)

    # Build label encoders with same fit as training
    encoders = {}
    encoded_merged = raw_merged.copy()
    for col in ['playerID', 'teamID', 'awayTeamID', 'position']:
        le = LabelEncoder()
        encoded_merged[col] = le.fit_transform(encoded_merged[col].astype(str))
        encoders[col] = le

    # Look up encoded player ID
    player_mask = raw_merged['playerID'] == raw_player_id
    if not player_mask.any():
        print(f"Player '{args.player_name}' has no game data.")
        return
    encoded_player_id = encoded_merged.loc[player_mask, 'playerID'].iloc[0]

    # Look up encoded opp ID
    if opp_name_clean not in encoders['awayTeamID'].classes_:
        print(f"Opponent '{args.opp_name}' not found. Available: {list(encoders['awayTeamID'].classes_)}")
        return
    encoded_opp_id = int(encoders['awayTeamID'].transform([opp_name_clean])[0])

    # Get player's last SEQUENCE_LENGTH games from encoded+normalized data
    _normalize_features(encoded_merged)
    player_games = (
        encoded_merged[encoded_merged['playerID'] == encoded_player_id]
        .sort_values('gameDate')
        .tail(SEQUENCE_LENGTH)
    )

    if len(player_games) < SEQUENCE_LENGTH:
        print(f"Not enough games for '{args.player_name}', found {len(player_games)}, need {SEQUENCE_LENGTH}")
        return

    arr = player_games[FEATURE_COLS].to_numpy(dtype=np.float32)

    inputs = {
        'input_player': np.array([[encoded_player_id] * SEQUENCE_LENGTH], dtype=np.int32),
        'input_posn':   player_games['position'].to_numpy().reshape(1, SEQUENCE_LENGTH).astype(np.int32),
        'input_team':   player_games['teamID'].to_numpy().reshape(1, SEQUENCE_LENGTH).astype(np.int32),
        'input_opp':    player_games['awayTeamID'].to_numpy().reshape(1, SEQUENCE_LENGTH).astype(np.int32),
        'input_stats':  arr.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS)),
        'input_ctx_player': np.array([[encoded_player_id]], dtype=np.int32),
        'input_ctx_opp':    np.array([[encoded_opp_id]], dtype=np.int32),
        'input_isHome':     np.array([[args.is_home]], dtype=np.float32),
    }

    predictions = model.predict(inputs)
    target_names = ['Fantasy_Pts', 'ThreePT_successes', 'FG_successes', 'FT_successes',
                    'REB', 'AST', 'BLK', 'STL', 'TO']

    print(f"\nPredictions for {args.player_name} vs {args.opp_name} ({'Home' if args.is_home else 'Away'}):")
    for name, pred in zip(target_names, predictions):
        print(f"  {name}: {pred[0][0]:.2f}")

if __name__ == "__main__":
    main()