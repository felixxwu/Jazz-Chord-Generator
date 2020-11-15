import operator
import json
import numpy as np
import random
from chords.chord import Chord

class NGrams:
    def __init__(self, sequences):
        self.sequences = sequences
        self.ngrams = {}
        self.total = 0
        self.sortedNgrams = None

        # print(f'building n = {N}', flush=True)
        # self.build(N)

    def getRandom(self):
        return random.choice(random.choice(self.sequences))

    def getNext(self, words, exclude = []):
        if len(words) != self.N - 1: return None, None

        probs = self.getProbs(words)
        if len(probs) <= len(exclude): 
            print("All possible chords exhausted, generating random...")
            return self.getRandom(), 0

        # print("top three options:")
        # for p in probs[:3]: print(p)
        
        if len(probs) == 0:
            # print("COULDN'T FIND SEQUENCE")
            return None, None
        
        r = random.random()
        for prob in probs:
            r -= prob[1]
            if r < 0:

                for excludeChord in exclude:
                    if json.dumps(prob[0]) == json.dumps(excludeChord.getJson()):
                        return self.getNext(words, exclude)

                return prob[0], prob[1]

        return None, None
        # chordlist, problist = zip(*probs)
        # choice =  np.random.choice(chordlist, p = problist)
        # print(sum(problist))
        # return choice

    def add(self, gram):
        dumped = json.dumps(gram)
        if dumped in self.ngrams:
            self.ngrams[dumped] += 1
        else:
            self.ngrams[dumped] = 1

        self.total += 1

    def sort(self):
        self.sortedNgrams = sorted(self.ngrams.items(), key=operator.itemgetter(1), reverse=True)

    def startsWith(self, gram, start):
        for a, b in zip(gram, start):
            if a != b:
                return False
        return True

    def getProbs(self, words):
        filtered = [
            gram
            for gram
            in self.sortedNgrams
            if self.startsWith(json.loads(gram[0]), words)
        ]
        count = 0
        for item in filtered:
            count += item[1]
        last = lambda x: json.loads(x)[-1]
        return list(map(lambda x: (last(x[0]), x[1] / count), filtered))

    def build(self, N):
        self.N = N
        self.ngrams = {}
        for sequence in self.sequences:
            for i in range(len(sequence) - N):
                gram = [sequence[i + n] for n in range(N)]
                self.add(gram)

        self.sort()
        # print(self.sortedNgrams[:10])

    def buildUsingToSymbol(self):
        for song in self.sequences:
            for i in range(len(song) - self.N):
                gram = [song[i + n].toSymbol(keyLess=True) for n in range(self.N)]
                self.add(gram)

        self.sort()
        # print(self.sortedNgrams[:10])

    
        