import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TunableLayer(nn.Module):
    def __init__(self, name, input_nodes, output_nodes, activation):
        super().__init__()
        self.name = name
        self.layer = nn.Linear(input_nodes, output_nodes)
        self.activation = \
            F.relu if activation == 'relu' or activation == 0 else\
            torch.sigmoid if activation == 'sigmoid' or activation == 1 else\
            torch.tanh if activation == 'tanh' or activation == 2 else\
            None

    def forward(self, x):
        x = self.activation(self.layer(x))
        return x

class TunableModel(nn.Module):
    def __init__(self, input_dim, output_dim, params):
        super().__init__()
        
        layers = nn.ModuleList([])
        layers.append(
            TunableLayer('Input', params['input_dim'], params['start_nodes'], params['activation'])
        )
        nodes = params['start_nodes']
        for i in range(params['num_layers']):
            next_nodes = max(int(nodes * params['decay_rate']), params['output_dim'])
            layers.append(
                TunableLayer(f'Layer_{i}', nodes, next_nodes, params['activation'])
            )
            nodes = next_nodes
        layers.append(
            TunableLayer('Output', nodes, params['output_dim'], 'sigmoid')
        )

        self.layers = layers
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x
