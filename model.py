SEQUENCE_LENGTH = 20

from util.model_lstm import Input_Node, Embedded_Node, Concatenate_Node, LSTM_Node, Dense_Node, Layer_Tree, Model_Manager
from util.data_preprocessing import generate_all_sequences
from process import get_embed

embedding_sizes = get_embed()

# Can use list comprehension
# inputs = ['playerID', 'position', 'teamID', 'awayTeamID', 'isHome']
input_nodes = {
    'playerID_input'    : Input_Node('playerID_input',   'int32', (SEQUENCE_LENGTH,)),
    'position_input'    : Input_Node('position_input',   'int32', (SEQUENCE_LENGTH,)),
    'teamID_input'      : Input_Node('teamID_input',     'int32', (SEQUENCE_LENGTH,)),
    'awayTeamID_input'  : Input_Node('awayTeamID_input', 'int32', (SEQUENCE_LENGTH,)),
    'isHome_input'      : Input_Node('isHome_input',     'int32', (SEQUENCE_LENGTH,))
}
isMasking = False
embedded_nodes = {
    'playerID_embedded': Embedded_Node(
        input_nodes.get('playerID_input'),
        'playerID_embedded',
        SEQUENCE_LENGTH,
        int(embedding_sizes.get('playerID')),
        50,
        isMasking
    ),
    'position_embedded': Embedded_Node(
        input_nodes.get('position_input'),
        'position_embedded',
        SEQUENCE_LENGTH,
        int(embedding_sizes.get('position')),
        50,
        isMasking
    ),
    'teamID_embedded': Embedded_Node(
        input_nodes.get('teamID_input'),
        'teamID_embedded',
        SEQUENCE_LENGTH,
        int(embedding_sizes.get('teamID')),
        50,
        isMasking
    ),
    'awayTeamID_embedded': Embedded_Node(
        input_nodes.get('awayTeamID_input'),
        'awayTeamID_embedded',
        SEQUENCE_LENGTH,
        int(embedding_sizes.get('awayTeamID')),
        50,
        isMasking
    ),
}
concatenate_node = Concatenate_Node(
    [
        embedded_nodes['playerID_embedded'],
        embedded_nodes['position_embedded'],
        embedded_nodes['teamID_embedded'],
        embedded_nodes['awayTeamID_embedded'],
        input_nodes['isHome_input']
    ],
    'concatenated_features'
)
lstm_node = LSTM_Node(concatenate_node, 'lstm', 64)
hidden_node = Dense_Node(lstm_node, 'hidden', 32, 'relu')
output_nodes = [
    Dense_Node(hidden_node, 'Fantasy_pts_output',       1, 'linear'),    
    Dense_Node(hidden_node, 'ThreePT_successes_output', 1, 'linear'),
    Dense_Node(hidden_node, 'FG_successes_output',      1, 'linear'),
    Dense_Node(hidden_node, 'FT_successes_output',      1, 'linear'),
    Dense_Node(hidden_node, 'REB_output',               1, 'linear'),
    Dense_Node(hidden_node, 'AST_output',               1, 'linear'),
    Dense_Node(hidden_node, 'BLK_output',               1, 'linear'),
    Dense_Node(hidden_node, 'STL_output',               1, 'linear'),
    Dense_Node(hidden_node, 'TO_output' ,               1, 'linear'),
]     


layer_tree = Layer_Tree(output_nodes)
model_manager = Model_Manager('NBA_Prediction_model', layer_tree)
model_manager.create_model()
model_manager.compile()
model_manager.print_model()

# Usage:
SEQUENCES_NP_DATA, SEQUENCES_NP_LABEL = generate_all_sequences(
    merged_pd_data, SEQUENCE_LENGTH, 'playerID', feature_list, label_list
)

def get_info():
    return SEQUENCES_NP_DATA, SEQUENCES_NP_LABEL