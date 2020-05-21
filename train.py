print('Parsing XML...')
import parseAllFiles

print('Importing libraries...', flush=True)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os

from torch.autograd import Variable
from dataset.dataset import Dataset
from networks.networkbuilder import NetworkBuilder

window_size = 5
early_stopping_ratio = 0.5
hidden_layers = [500]
num_epochs = 5000
batch_size = 100
learning_rate = 0.0005
train_val_split = 0.85
gpu = torch.cuda.is_available()

# architecture = 'FNN'
# architecture = 'CNN/12/12/12'
architecture = 'LSTM/1'
# EXTRAS:
# CNN/kernels/flip/stride
# W2V/iterations/size/window
# LSTM/bias


print('Preparing data...', flush=True)
dataset = Dataset(window_size, train_val_split, batch_size, lstm=(architecture.split('/')[0] == 'LSTM'))
x_train, y_train, x_val, y_val = dataset.prepareData()
num_classes = len(y_train[0][0])
input_size = len(x_train[0][0])

os.system('mkdir output')
filename = f"output/{architecture.replace('/', '-')}.png"
txtout = f"output/{architecture.replace('/', '-')}.txt"
print("Plotting to file:", filename, flush=True)
plot_title = f"{architecture} W{window_size} H{hidden_layers} B{batch_size} L{learning_rate}"

print("Input size:", input_size, flush=True)
print("Output size:", num_classes, flush=True)
print("Architecture:", architecture, flush=True)
print("x_train:", np.array(x_train).shape, flush=True)
print("y_train:", np.array(y_train).shape, flush=True)
print("x_val:", np.array(x_val).shape, flush=True)
print("y_val:", np.array(y_val).shape, flush=True)
print("Using GPU:", gpu, flush=True)
print("Hidden layers:", hidden_layers, flush=True)
print("Defining network...", flush=True)

networkBuilder = NetworkBuilder(architecture, input_size, hidden_layers, num_classes, batch_size, window_size)
net = networkBuilder.getNetwork()
if gpu:
    net.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

val_history = []
train_history = []

def shouldEarlyStop():
    # return False
    max_index = val_history.index(max(val_history))
    return max_index < len(val_history) * early_stopping_ratio - 30

print('Training...', flush=True)
for epoch in range(num_epochs):
    running_loss = 0.0
    for index, (inputs, labels) in enumerate(zip(x_train, y_train)):
        print(f"{index}/{len(x_train)}", end='\r', flush=True)
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        outputs = net(inputs)
        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        
        running_loss += loss.item()
        
    val_acc = dataset.getAccuracy(net, x_val, y_val, gpu)
    train_acc = dataset.getAccuracy(net, x_train, y_train, gpu)
    print('Epoch:', epoch + 1, 'training_loss:', round(running_loss / len(x_train), 4), 'val_acc:', round(val_acc, 4), flush=True)
    val_history += [val_acc]
    train_history += [train_acc]

    plt.clf()
    plt.plot(train_history, label='train_acc')
    plt.plot(val_history, label='val_acc')
    plt.legend(loc='lower right')
    plt.ylabel('val accuracy / similarity')
    plt.xlabel(" last updated(" + str(datetime.datetime.now().time()) + ")")
    plt.title(plot_title + " hi" + str(round(max(val_history), 4)))
    plt.savefig(filename)

    f = open(txtout, "w")
    f.write(str(sys.argv))
    f.write('\n')
    f.write(str(round(max(val_history), 4)))
    f.close()

    torch.save(net.state_dict(), f"output/{architecture.replace('/', '-')}.pkl")
    
    if shouldEarlyStop():
        print("Stopping early, best was", round(max(val_history), 4), flush=True)
        break




                
