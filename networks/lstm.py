import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, forget):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_layers[0])
        self.forget = forget
        print('Forget bias:', forget, flush=True)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(self.forget)

        self.hiddenLayers = nn.ModuleList([
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], num_classes),
        ])
    
    def forward(self, x):
        x = x.view(len(x), 1, -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.view(len(x), -1)

        for layer in self.hiddenLayers:
            lstm_out = layer(lstm_out)

        return lstm_out