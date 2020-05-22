import torch

class Evaluate:
    def __init__(self):
        pass

    def accuracy(self, network, x_val, y_val, gpu):
        correct = 0
        total = 0
        for index, (inputs, labels) in enumerate(zip(x_val, y_val)):
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            if gpu:
                network.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = network(inputs)
            for i, output in enumerate(outputs):
                correct += self.evalOneHot(output.tolist(), labels[i].tolist())
                total += 1
            print(f'Calculating accuracy: {index}/{len(x_val)}', end='\r')

        return correct / total

    def evalOneHot(self, output, label):
        maxOutput = output.index(max(output))
        maxLabel = label.index(max(label))
        return maxOutput == maxLabel
