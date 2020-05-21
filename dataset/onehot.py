from dataset.readData import rootAndDegrees as readData
from chords.chord import Chord

def vectorise(method):
    return [
        [
            method(chord)
            for chord in seq
        ]
        for seq in readData()
    ]

def oneHotChords():
    unique = []
    for seq in readData():
        for chord in seq:
            symbol = Chord(chord).toSymbol(keyLess=True)
            if symbol not in unique:
                unique += [symbol]

    def method(chord):
        symbol = Chord(chord).toSymbol(keyLess=True)
        index = unique.index(symbol)
        vector = [0.0] * len(unique)
        vector[index] = 1.0
        return vector

    vec = vectorise(method)
    return vec, unique