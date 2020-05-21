import json

def readData(modifier):
    data = json.load(open('dataset/parsed.json'))
    represent = lambda chord: {'root': chord['root'], 'components': modifier(chord['degrees'])}

    seqs = []
    for key in data.keys():
        song = data[key]
        seq = []
        for measureKey in song.keys():
            measure = song[measureKey]
            for chord in measure:
                seq += [represent(chord)]
        seqs += [seq]
    
    return seqs


def rootAndDegrees():
    return readData(lambda x: x)

def rootAndDegreesBasic():
    def modifier(degrees):
        no7 = degrees[:]
        if 1 in no7: no7.remove(1)
        if 2 in no7: no7.remove(2)
        if 3 not in no7 and 4 not in no7: no7 += [4]
        if 5 in no7: no7.remove(5)
        if 6 in no7: no7.remove(6)
        if 8 in no7: no7.remove(8)
        if 9 in no7: no7.remove(9)
        if 10 in no7: no7.remove(10)
        if 11 in no7: no7.remove(11)
        no7.sort()
        return no7

    return readData(modifier)

def rootAndDegreesOnly7():
    def modifier(degrees):
        no7 = degrees[:]
        if 1 in no7: no7.remove(1)
        if 2 in no7: no7.remove(2)
        if 3 not in no7 and 4 not in no7: no7 += [4]
        if 3 in no7 and 4 in no7: no7.remove(3)
        if 5 in no7: no7.remove(5)
        if 6 in no7: no7.remove(6)
        if 8 in no7: no7.remove(8)
        if 9 in no7: no7.remove(9)
        no7.sort()
        return no7

    return readData(modifier)
