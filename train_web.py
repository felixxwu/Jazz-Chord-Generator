"""Trains browser-sized FNN, CNN and LSTM models on each chord representation from the paper (§5.2)
and exports them to web/data/models. Run export_web.py first. Needs torch and gensim.

Usage: python train_web.py <rep> [model ...]   e.g. python train_web.py ene fnn lstm
       python train_web.py encodings           (writes web/data/encodings.json, run once first)
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn

DATA = 'web/data/'
WINDOW = 4
REPS = ['1he', 'ene', 'wne', 'nne', '2he', 'w2v']
HIDDEN = {'fnn': 10000, 'cnn': 2000, 'lstm': 1069}
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

vocab = json.load(open(DATA + 'vocab.json'))
songs = json.load(open(DATA + 'songs.json'))['seqs']
split = int(len(songs) * 0.85)
train_songs, val_songs = songs[:split], songs[split:]
V = len(vocab)


def word2vec():
    from gensim.models import Word2Vec
    model = Word2Vec([[str(c) for c in s] for s in train_songs], vector_size=100, window=4, epochs=100,
                     sg=1, min_count=1, seed=1, workers=1)
    return np.array([model.wv[str(i)] if str(i) in model.wv else np.zeros(100) for i in range(V)], dtype=np.float32)


def build_encodings():
    """Each chord's vector in every representation, as described in §5.2 of the paper."""
    types = sorted({tuple(v['comps']) for v in vocab})
    tables = {rep: [] for rep in REPS}
    for i, v in enumerate(vocab):
        root, comps = v['root'], v['comps']
        tables['1he'].append([i])
        tables['ene'].append(sorted({root} | {root + c for c in comps}))
        tables['wne'].append(sorted({root} | {(root + c) % 12 for c in comps}))
        tables['nne'].append([root] + [11 + c for c in comps])
        tables['2he'].append([root, 12 + types.index(tuple(comps))])
    dims = {'1he': V, 'ene': 24, 'wne': 12, 'nne': 23, '2he': 12 + len(types), 'w2v': 100}
    w2v = word2vec()
    tables['w2v'] = np.round(w2v, 4).tolist()
    json.dump({rep: {'dim': dims[rep], 'dense': rep == 'w2v', 'vectors': tables[rep]} for rep in REPS},
              open(DATA + 'encodings.json', 'w'), separators=(',', ':'))


def encoding_matrix(rep):
    enc = json.load(open(DATA + 'encodings.json'))[rep]
    if enc['dense']:
        return torch.tensor(enc['vectors'], dtype=torch.float32)
    m = torch.zeros(V, enc['dim'])
    for i, ones in enumerate(enc['vectors']):
        m[i, ones] = 1
    return m


def windows(seqs):
    """Every position after the first chord, with the last WINDOW chords as context.
    Shorter contexts at the start of a song are repeated to fill the window, like the web app does."""
    xs, ys = [], []
    for s in seqs:
        for i in range(1, len(s)):
            ctx = s[max(0, i - WINDOW):i]
            xs.append([ctx[(len(ctx) - WINDOW + j) % len(ctx)] for j in range(WINDOW)])
            ys.append(s[i])
    return torch.tensor(xs), torch.tensor(ys)


class FNN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(WINDOW * dim, hidden), nn.Dropout(0.2), nn.BatchNorm1d(hidden), nn.ReLU(),
                                    nn.Linear(hidden, V))

    def forward(self, x):
        return self.layers(x.flatten(1))


class CNN(nn.Module):
    """Convolution across time with kernel size 3, the best CNN setup in §6.6."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3)
        self.layers = nn.Sequential(nn.Linear(dim * (WINDOW - 2), hidden), nn.Dropout(0.2), nn.BatchNorm1d(hidden),
                                    nn.ReLU(), nn.Linear(hidden, V))

    def forward(self, x):
        return self.layers(torch.relu(self.conv(x.transpose(1, 2))).flatten(1))


class LSTM(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        # Forget gate bias of 1.0 worked best (§6.7.2). PyTorch adds two bias vectors, so each gets half.
        for bias in (self.lstm.bias_ih_l0, self.lstm.bias_hh_l0):
            bias.data[hidden:2 * hidden].fill_(0.5)
        self.layers = nn.Sequential(nn.Dropout(0.2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, V))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.layers(out[:, -1])


def evaluate(net, enc, x, y, w2v):
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(enc[x[i:i + 2000]]).argmax(1) for i in range(0, len(x), 2000)])
    net.train()
    sim = nn.functional.cosine_similarity(w2v[pred], w2v[y], dim=1, eps=1e-8)
    return (pred == y).float().mean().item(), sim.mean().item()


def train(rep, model):
    torch.manual_seed(0)
    enc = encoding_matrix(rep).to(DEVICE)
    w2v = encoding_matrix('w2v').to(DEVICE)
    x_train, y_train = (t.to(DEVICE) for t in windows(train_songs))
    x_val, y_val = (t.to(DEVICE) for t in windows(val_songs))
    net = {'fnn': FNN, 'cnn': CNN, 'lstm': LSTM}[model](enc.shape[1], HIDDEN[model]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.0005)
    best, best_state, since_best = (0, 0), None, 0
    for epoch in range(300):
        for idx in torch.randperm(len(x_train), device=DEVICE).split(100):
            opt.zero_grad()
            nn.functional.cross_entropy(net(enc[x_train[idx]]), y_train[idx]).backward()
            opt.step()
        acc, sim = evaluate(net, enc, x_val, y_val, w2v)
        print(f'{model}-{rep} epoch {epoch + 1} acc {acc:.4f} sim {sim:.4f}', flush=True)
        if acc > best[0]:
            best, since_best = (acc, sim), 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since_best += 1
            if since_best >= 15:
                break
    net.load_state_dict(best_state)
    net.cpu()
    export(f'{DATA}models/{model}-{rep}', net, model, {'accuracy': best[0], 'similarity': best[1]})


def export(path, net, model, metrics):
    """Writes 2D weights as int8 with one float32 scale per row, everything else as float32."""
    sd = net.state_dict()
    if model == 'fnn':
        tensors = [('w0', sd['layers.0.weight']), ('b0', sd['layers.0.bias'])]
        bn, out = 'layers.2', 'layers.4'
    elif model == 'cnn':
        w = sd['conv.weight']
        tensors = [('conv_w', w.permute(0, 2, 1).reshape(w.shape[0], -1)), ('conv_b', sd['conv.bias']),
                   ('w0', sd['layers.0.weight']), ('b0', sd['layers.0.bias'])]
        bn, out = 'layers.2', 'layers.4'
    else:
        tensors = [('wi', sd['lstm.weight_ih_l0']), ('wh', sd['lstm.weight_hh_l0']),
                   ('b', sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0'])]
        bn, out = 'layers.1', 'layers.3'
    tensors += [('bn_w', sd[bn + '.weight']), ('bn_b', sd[bn + '.bias']), ('bn_mean', sd[bn + '.running_mean']),
                ('bn_var', sd[bn + '.running_var']), ('out_w', sd[out + '.weight']), ('out_b', sd[out + '.bias'])]

    manifest, blob = [], bytearray()
    for name, t in tensors:
        t = t.detach().float().numpy()
        entry = {'name': name, 'shape': list(t.shape), 'offset': len(blob)}
        if t.ndim == 2:
            scale = np.abs(t).max(axis=1) / 127 + 1e-12
            blob += scale.astype('<f4').tobytes()
            blob += np.round(t / scale[:, None]).astype(np.int8).tobytes()
            entry['type'] = 'int8'
        else:
            blob += t.astype('<f4').tobytes()
            entry['type'] = 'f32'
        blob += b'\0' * (-len(blob) % 4)
        manifest.append(entry)
    open(path + '.bin', 'wb').write(bytes(blob))
    json.dump({'tensors': manifest, **metrics}, open(path + '.json', 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    if sys.argv[1] == 'encodings':
        build_encodings()
    else:
        for model in sys.argv[2:] or ['fnn', 'cnn', 'lstm']:
            train(sys.argv[1], model)
