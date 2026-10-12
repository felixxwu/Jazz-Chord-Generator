// The paper's next-chord models. Chords come in as vocabulary indices and are turned into
// the chosen representation's vector (§5.2) before going into the network.

export const WINDOW = 4

export function loadWeights(manifest, buffer) {
  const tensors = {}
  for (const { name, shape, offset, type } of manifest) {
    if (type === 'f32') {
      tensors[name] = new Float32Array(buffer, offset, shape[0])
    } else {
      const [rows, cols] = shape
      tensors[name] = {
        rows,
        cols,
        scale: new Float32Array(buffer, offset, rows),
        q: new Int8Array(buffer, offset + rows * 4, rows * cols),
      }
    }
  }
  return tensors
}

// Most representations are a few ones in a long vector, so vectors are kept as (index, value) pairs.
export function encoder({ dense, vectors }) {
  const codes = vectors.map(v =>
    dense ? { idx: v.map((_, i) => i), val: v } : { idx: v, val: v.map(() => 1) },
  )
  return chord => codes[chord]
}

// Sums one row of weights against several encoded chords placed side by side in the input.
function rowDot(m, row, blocks, width) {
  let sum = 0
  const base = row * m.cols
  blocks.forEach(({ idx, val }, b) => {
    for (let n = 0; n < idx.length; n++) sum += m.q[base + b * width + idx[n]] * val[n]
  })
  return sum * m.scale[row]
}

function matVec(m, x, bias) {
  const out = new Float32Array(m.rows)
  for (let r = 0; r < m.rows; r++) {
    let sum = 0
    const base = r * m.cols
    for (let c = 0; c < m.cols; c++) sum += m.q[base + c] * x[c]
    out[r] = sum * m.scale[r] + (bias ? bias[r] : 0)
  }
  return out
}

const sigmoid = x => 1 / (1 + Math.exp(-x))

function head(w, x) {
  const normed = x.map((v, i) => Math.max(0, ((v - w.bn_mean[i]) / Math.sqrt(w.bn_var[i] + 1e-5)) * w.bn_w[i] + w.bn_b[i]))
  return matVec(w.out_w, normed, w.out_b)
}

// Shorter progressions are repeated to fill the window, as jazz progressions usually loop.
function lastWindow(seq) {
  return Array.from({ length: WINDOW }, (_, j) => seq[(((seq.length - WINDOW + j) % seq.length) + seq.length) % seq.length])
}

function fnn(w, codes, width) {
  const hidden = new Float32Array(w.w0.rows)
  for (let r = 0; r < w.w0.rows; r++) hidden[r] = rowDot(w.w0, r, codes, width) + w.b0[r]
  return head(w, hidden)
}

function cnn(w, codes, width) {
  const channels = w.conv_w.rows
  const steps = WINDOW - 2
  const conv = new Float32Array(channels * steps)
  for (let o = 0; o < channels; o++) {
    for (let t = 0; t < steps; t++) {
      conv[o * steps + t] = Math.max(0, rowDot(w.conv_w, o, codes.slice(t, t + 3), width) + w.conv_b[o])
    }
  }
  return head(w, matVec(w.w0, conv, w.b0))
}

function lstm(w, codes) {
  const H = w.wh.cols
  let h = new Float32Array(H)
  const c = new Float32Array(H)
  for (const code of codes) {
    const gates = matVec(w.wh, h, w.b)
    for (let j = 0; j < 4 * H; j++) gates[j] += rowDot(w.wi, j, [code], 0)
    const next = new Float32Array(H)
    for (let j = 0; j < H; j++) {
      const i = sigmoid(gates[j])
      const f = sigmoid(gates[H + j])
      const g = Math.tanh(gates[2 * H + j])
      const o = sigmoid(gates[3 * H + j])
      c[j] = f * c[j] + i * g
      next[j] = o * Math.tanh(c[j])
    }
    h = next
  }
  return head(w, h)
}

export function networkLogits(model, w, encode, width, seq) {
  const codes = lastWindow(seq).map(encode)
  return { fnn, cnn, lstm }[model](w, codes, width)
}

export function softmax(logits, temperature = 1) {
  const max = Math.max(...logits)
  const exps = Array.from(logits, v => Math.exp((v - max) / temperature))
  const total = exps.reduce((a, b) => a + b, 0)
  return exps.map(v => v / total)
}

// Katz-style back-off from the paper: shorten the context until at least k examples are found.
export function ngram(songs, seq, k, maxN, vocabSize) {
  for (let n = Math.min(seq.length, maxN - 1); n >= 0; n--) {
    const context = seq.slice(seq.length - n)
    const counts = new Float64Array(vocabSize)
    let total = 0
    for (const song of songs) {
      for (let i = n; i < song.length; i++) {
        let match = true
        for (let j = 0; j < n && match; j++) match = song[i - n + j] === context[j]
        if (match) {
          counts[song[i]]++
          total++
        }
      }
    }
    if (total >= k || n === 0) return { probs: Array.from(counts, v => v / total), n: n + 1, examples: total }
  }
}

export function descendingFifth(vocab, seq) {
  const root = (vocab[seq[seq.length - 1]].root + 5) % 12
  const best = vocab.reduce((best, v, i) => (v.root === root && (best < 0 || v.count > vocab[best].count) ? i : best), -1)
  const probs = new Array(vocab.length).fill(0)
  probs[best] = 1
  return probs
}
