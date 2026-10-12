"""Exports the chord vocabulary and songs to web/data for the browser app; train_web.py builds the models."""
import json
from chords.chord import Chord

OUT = 'web/data/'

# Sorted so the vocabulary order matches the original checkpoints in output/, which were trained
# on Windows where the XML files were listed alphabetically.
data = json.load(open('dataset/parsed.json'))
composers = json.load(open('dataset/composers.json'))
names = sorted(data)

vocab, index, songs = [], {}, []
for name in names:
    seq = []
    for measure in data[name].values():
        for c in measure:
            chord = Chord({'root': c['root'], 'components': c['degrees']})
            symbol = chord.toSymbol(keyLess=True)
            if symbol not in index:
                index[symbol] = len(vocab)
                suffix = symbol[len(str(chord.root)):]
                vocab.append({'root': chord.root, 'comps': chord.components, 'suffix': suffix, 'count': 0})
            vocab[index[symbol]]['count'] += 1
            seq.append(index[symbol])
    songs.append(seq)

json.dump(vocab, open(OUT + 'vocab.json', 'w'), separators=(',', ':'))
json.dump({
    'names': [n[:-4] for n in names],
    'composers': [composers.get(n) or '' for n in names],
    'seqs': songs,
}, open(OUT + 'songs.json', 'w'), separators=(',', ':'))


print(len(vocab), 'chords,', len(songs), 'songs')
