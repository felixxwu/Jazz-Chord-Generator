import random
import numpy as np
from dataset.onehot import oneHotChords
from dataset.evaluate import Evaluate
from dataset.sustain import sustain

class Dataset:
    def __init__(self, window, split, batchSize, lstm=False):
        self.window = window
        self.split = split
        self.batchSize = batchSize
        self.lstm = lstm

        self.evaluationClass = Evaluate()

    def prepareData(self):
        x_seqs = self.getXseqs()
        y_seqs = self.getYseqs()

        if self.lstm:
            return self.createXYLSTM(x_seqs, y_seqs, self.window, self.split)
        
        return self.createXY(x_seqs, y_seqs, self.window, self.split, self.batchSize)

    def getAccuracy(self, network, x, y, gpu):
        return self.evaluationClass.accuracy(network, x, y, gpu)

    def getYseqs(self):
        y_seqs, unique = oneHotChords()
        return y_seqs

    def getXseqs(self):
        x_seqs, unique = oneHotChords()
        return x_seqs

    # size [n, m] -> [n/4, 4]
    def tupletify(self, seqs, size):
        tuplets = []
        for seq in seqs:
            i = size
            while i < len(seq):
                tuplets += [seq[(i - size):i]]
                i += size
        return tuplets

    @staticmethod
    def splitTrainVal(seqs, ratio):
        splitAt = int(len(seqs) * ratio)
        train = seqs[0:splitAt]
        val = seqs[splitAt:]
        return train, val

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def createXY(self, x_seqs, y_seqs, n, split, batchSize):
        x = []
        y = []
        for (x_seq, y_seq) in list(zip(x_seqs, y_seqs)):
            i = n
            while i < len(x_seq):
                x += [self.flatten(x_seq[(i - n):i])]
                y += [y_seq[i]]
                i += 1
        assert len(x) == len(y)
        zipped = list(zip(x, y))
        random.seed(69)
        random.shuffle(zipped)
        assert split > 0
        assert split < 1
        splitAt = int(len(x) * split)

        x_train, y_train = list(zip(*zipped[0:splitAt]))
        x_val, y_val = list(zip(*zipped[splitAt:]))
        
        x_train = self.batchify(x_train, batchSize)
        y_train = self.batchify(y_train, batchSize)
        x_val = self.batchify(x_val, batchSize)
        y_val = self.batchify(y_val, batchSize)

        return x_train, y_train, x_val, y_val

    def createXYLSTM(self, x_seqs, y_seqs, n, split):
        x = []
        y = []
        for (x_seq, y_seq) in list(zip(x_seqs, y_seqs)):
            x += [x_seq[:-1]]
            y += [y_seq[1:]]

        x_train, x_val = self.splitTrainVal(x, split)
        y_train, y_val = self.splitTrainVal(y, split)

        return x_train, y_train, x_val, y_val

    def batchify(self, array, size):
        out = []
        i = size
        while i < len(array):
            out += [array[(i - size):i]]
            i += size
        return out

    
        
