from networks.fnn import FNN
from networks.cnn import CNN
from networks.lstm import LSTM

class NetworkBuilder:
    def __init__(self, architecture, input_size, hidden_layers, num_classes, batch_size, context_window):
        self.architecture = architecture.split('/')
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.context_window = context_window

    def getNetwork(self):
        if self.architecture[0] == 'FNN':
            return FNN(self.input_size, self.hidden_layers, self.num_classes)

        if self.architecture[0] == 'CNN':
            kernels = int(self.architecture[1])
            flip = int(self.architecture[2])
            stride = int(self.architecture[3])
            return CNN(self.batch_size, self.input_size, self.hidden_layers, self.num_classes, kernels, flip, stride, self.context_window)

        if self.architecture[0] == 'LSTM':
            forget = float(self.architecture[1])
            return LSTM(self.input_size, self.hidden_layers, self.num_classes, forget)