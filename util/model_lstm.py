from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Layer
from keras.utils import plot_model
from enum import Enum
import numpy as np

class Layer_Node:
    def __init__(self, parent_nodes, name):
        self.parent_nodes = parent_nodes
        self.name = name
        self.layer = None

    def create_layer(self):
        raise NotImplementedError("Subclasses must implement create_layer()")
    
    def get_parent_layers(self):
        parent_layers = []
        for parent in self.parent_nodes:
            if   parent is None:       parent_layers.append(None)
            elif parent.layer is None: parent_layers(parent.create_layer())
            else:                      parent_layers(parent.layer)
        return parent_layers

class Input_Node(Layer_Node):
    def __init__(self, 
                 name, data_type, shape):
        super().__init__(None, name)
        self.data_type = data_type
        self.shape = shape

    def create_layer(self):
        self.layer = Input(
            shape=self.shape, 
            dtype=self.data_type, 
            name=self.name
        )

class Embedded_Node(Layer_Node):
    def __init__(self, parent_nodes, name,
                 sequence_len, input_dim, output_dim, isMasking=False):
        super().__init__(parent_nodes, name)
        self.sequence_len = sequence_len
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.isMasking    = isMasking

    def create_layer(self):
        parent_layers = self.get_parent_layers()
        self.layer = Embedding(
            name         = self.name,
            input_length = self.sequence_len,
            input_dim    = self.input_dim,
            output_dim   = self.output_dim,
            mask_zero    = self.isMasking
        )(parent_layers)

class LSTM_Node(Layer_Node):
    def __init__(self, parent_nodes, name,
                 units):
        super().__init__(parent_nodes, name)
        self.units  = units

    def create_layer(self):
        # Required for GPU accel
        activation = 'tanh', rec_activation = 'sigmoid'
        unroll = False, use_bias = True
        rec_dropout = 0

        parent_layers = self.get_parent_layers()
        self.layer = LSTM(
                self.units,                    
                name=self.name,
                activation=activation,   
                recurrent_activation=rec_activation,
                recurrent_dropout=rec_dropout,      
                unroll=unroll,       
                use_bias=use_bias   
        )(parent_layers)   
    
class Concatenate_Node(Layer_Node):
    def __init__(self, parent_nodes, name):
        super().__init__(parent_nodes, name)

    def create_layer(self):
        parent_layers = self.get_parent_layers()
        self.layer = Concatenate(
            axis=-1, 
            name=self.name
        )(parent_layers)

class Dense_Node(Layer_Node):
    def __init__(self, parent_nodes, name, 
                 units, activation):
        super().__init__(parent_nodes, name)
        self.units = units
        self.activaton = activation

    def create_layer(self):
        parent_layers = self.get_parent_layers()
        self.layer = Dense(
            self.units,
            activation = self.activation,
            name =       self.name
        )(parent_layers)

class Layer_Tree:
    def __init__(self, output_nodes):
        self.output_nodes  = output_nodes
        self.output_layers = set()
        self.input_layers  = set()

    def create_layers(self):
        for output_node in self.output_nodes:
            output_node.create_layer()
        self._gather_layers()
        
    def _gather_layers(self):
        def traverse(node):
            if node.parent_nodes is None:
                self.input_layers.add(node.layer)
            else:
                for parent in node.parent_nodes:
                    traverse(parent)
        
        for output_node in self.output_nodes:
            self.output_layers.add(output_node.layer)
            traverse(output_node)

class Model_Manager:
    def __init__(self, name, layer_tree):
        self.name = name
        self.layer_tree = layer_tree
        self.model = None

    # TODO: Use the layer_tree to construct the model
    def create_model(self):
        self.model = Layer( 
            inputs=  list(self.layer_tree.input_layers),
            outputs= list(self.layer_tree.output_layers),
            name=    self.name
        )

    def save(self, filePath):
        try:
            self.model.save(filePath)
        except Exception as e:
            print(f"Error saving model to {filePath}: {e}")

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer, loss)

    def print_model(self):
        plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)