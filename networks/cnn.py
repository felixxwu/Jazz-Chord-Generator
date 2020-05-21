import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    # FIRST HIDDEN LAYER GETS AUTOMATICALLY REPLACED BY SUITABLE CONV OUTPUT
    def __init__(self, batch_size, input_size, hidden_layers, num_classes, kernels, flip, stride, context_window):
        super(CNN, self).__init__()
        self.kernels = kernels
        self.flip = bool(flip)
        self.stride = stride
        self.input_size = input_size
        self.context_window = context_window
        representation_length = input_size // context_window
        hidden_layers[0] = representation_length * ((context_window - kernels + 1) // stride)
        self.shapes = [
            [batch_size, representation_length, context_window],
            [batch_size, -1]
        ]
        
        if self.flip:
            hidden_layers[0] = context_window * ((representation_length - kernels + 1) // stride)
            representation_length = context_window

        self.hiddenLayers = nn.ModuleList([
            nn.Conv1d(
                in_channels=representation_length,
                out_channels=representation_length,
                kernel_size=self.kernels,
                stride=self.stride
            ),
            nn.ReLU(),
        ])

        for i in range(1, len(hidden_layers)):
            prev = hidden_layers[i - 1]
            curr = hidden_layers[i]
            self.hiddenLayers.append(nn.Linear(prev, curr))
            self.hiddenLayers.append(nn.Dropout(0.2))
            self.hiddenLayers.append(nn.BatchNorm1d(curr))
            self.hiddenLayers.append(nn.ReLU())

        self.hiddenLayers.append(nn.Linear(hidden_layers[len(hidden_layers) - 1], num_classes))
            
        print('Kernels', kernels)
        print('stride', stride)
        print('input_size', input_size)
        print('context_window', context_window)
        print('representation_length', representation_length)
        print('Settings first hidden layer to:', hidden_layers[0])
    
    def forward(self, x):
        # print(self.shapes)
        # print('(x).shape', np.array(x).shape)
        # x = x.view(-1, self.input_size)
        for i, _ in enumerate(self.hiddenLayers):
            if len(self.shapes) > i and self.shapes[i] != None:
                # print('shaper', self.shapes[i])
                x = x.view(*self.shapes[i])

            if self.flip and i == 0:
                # print(x.shape)
                x = x.transpose(1,2)
                # print(x.shape)

            # print(i, self.hiddenLayers[i])
            # print(x.shape)
            x = self.hiddenLayers[i](x)
            # print(x.shape)
        return x