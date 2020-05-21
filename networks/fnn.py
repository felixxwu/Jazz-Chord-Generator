import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(FNN, self).__init__()
        self.hiddenLayers = nn.ModuleList([
            nn.Linear(input_size, hidden_layers[0]),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_layers[0]),
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
    
    def forward(self, x):
        for i, _ in enumerate(self.hiddenLayers):
            # print(i, self.hiddenLayers[i])
            # print(x.shape)
            x = self.hiddenLayers[i](x)
            # print(x.shape)
        return x