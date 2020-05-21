import torch

class Evaluate:
    def __init__(self):
        pass

    def accuracy(self, network, x_val, y_val, gpu):
        correct = 0
        total = 0
        for (inputs, labels) in zip(x_val, y_val):
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = network(inputs)
            for i, output in enumerate(outputs):
                correct += self.evalOneHot(output.tolist(), labels[i].tolist())
                total += 1

        return correct / total

    def evalOneHot(self, output, label):
        maxOutput = output.index(max(output))
        maxLabel = label.index(max(label))
        return maxOutput == maxLabel
