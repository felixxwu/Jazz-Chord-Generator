// UI for the chord generator. Like the paper, it puts the chord representation first: the same progression is fed
// to a network through each of six representations, following Figure 1.1: input progression → representation → model → next chord.
// Chords are stored as vocabulary indices relative to the key, so changing the key transposes everything.
import { loadWeights, encoder, networkLogits, softmax, ngram, descendingFifth, WINDOW } from './models.js'

const $ = id => document.getElementById(id)

const FLATS = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
const SHARPS = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
const SHARP_KEYS = new Set([2, 4, 6, 9, 11])
const BLACK_KEYS = new Set([1, 3, 6, 8, 10])
const ROMAN = ['I', '♭II', 'II', '♭III', 'III', 'IV', '♯IV', 'V', '♭VI', 'VI', '♭VII', 'VII']

// The dataset's own chord spellings, translated to how jazz musicians write them.
const PRETTY = {
  M7: 'maj7', M9: 'maj9', M13: 'maj13', 'M7(+#11)': 'maj7(#11)', 'M9(+#11)': 'maj9(#11)',
  mdim7: 'm7b5', mdim: 'dim', maug: 'm#5', mM7: 'm(maj7)', mM9: 'm(maj9)', 'm6(+9)': 'm6/9', 'm(+b13)': 'm(b13)',
  '(+9)': 'add9', '6(+9)': '6/9',
  aug: '+', aug7: '7#5', aug9: '9#5', augM7: 'maj7#5', 'aug7(+#9)': '7#5(#9)', 'aug7(+b9)': '7#5(b9)',
  dim7: '7b5', dim9: '9b5', 'dim7(+#9)': '7b5(#9)', 'dim7(+b9)': '7b5(b9)',
  '11sus': '7sus4', '11sus2': '9sus4', '11(+b9)sus': '7sus4(b9)', '11(+b13)sus': '7sus4(b13)',
}
const QUALITIES = ['maj7', '7', 'm7', 'm7b5', '6', '7(b9)', '', 'm', 'm6', '9', '13', '7sus4', '7#5', '7(#9)', '7(#11)', 'maj7(#11)', 'dim', 'm(maj7)']

const PRESETS = {
  'ii–V–I': 'ii7 V7 Imaj7',
  'Rhythm changes': 'Imaj7 vi7 ii7 V7',
  'Minor ii–V–i': 'iim7b5 III7(b9) vim6',
  Blues: 'I7 IV7 I7 I7 IV7 IV7 I7 VI7 ii7 V7 I7 V7',
  'Giant Steps': 'Imaj7 bIII7 bVImaj7 VII7 IIImaj7',
}

const MODELS = [
  { id: 'fnn', title: 'Feed-Forward Neural Network', short: 'FNN', section: '5.3.2', page: 26, desc: 'reads the chords side by side through a hidden layer of 10,000 units.' },
  { id: 'cnn', title: 'Convolutional Neural Network', short: 'CNN', section: '5.3.3', page: 27, desc: 'slides a filter over 3 chords at a time to pick up patterns in time, then a hidden layer of 2,000 units.' },
  { id: 'lstm', title: 'Long Short-Term Memory Network', short: 'LSTM', section: '5.3.4', page: 29, desc: 'reads the chords one after another, carrying a memory of 1,069 units between them.' },
]

const BASELINES = [
  { id: 'ngram', title: 'N-gram', section: '5.3.1', page: 25 },
  { id: 'fifths', title: 'descending fifths', section: '6.4.2', page: 38 },
]

const REPS = [
  { id: '1he', name: 'One-Hot Encoding', short: '1HE', section: '5.2.2', page: 22, desc: 'One position for each of the 464 chords in the dataset. It says nothing about which notes are in the chord.' },
  { id: 'ene', name: 'Explicit Note Encoding', short: 'ENE', section: '5.2.3', page: 22, desc: 'Laid out like a 24-key keyboard starting on the key note, with the chord’s notes switched on.' },
  { id: 'wne', name: 'Wrapped Note Encoding', short: 'WNE', section: '5.2.4', page: 23, desc: 'Explicit note encoding folded into a single octave of 12.' },
  { id: 'nne', name: 'Normalised Note Encoding', short: 'NNE', paper: true, section: '5.2.5', page: 23, desc: 'The root as one of 12 notes, then the other notes measured up from the root, so every seventh chord looks alike apart from its root.' },
  { id: '2he', name: 'Two-Hot Encoding', short: '2HE', paper: true, section: '5.2.6', page: 24, desc: 'The root as one of 12 notes, then the chord type as one of 53.' },
  { id: 'w2v', name: 'Chord Embeddings (Word2Vec)', short: 'W2V', section: '5.2.7', page: 24, desc: '100 learned numbers per chord. Chords that turn up in similar places get similar vectors.' },
]

// Accuracy (%) and similarity reported in the paper (Tables 6.1 and 6.2).
const PAPER = {
  random: [0.22, 0.06], ngram: [37.47, 0.556], fifths: [41.27, 0.585],
  fnn: { '1he': [44.08, 0.598], w2v: [43.58, 0.595], '2he': [43.43, 0.592], nne: [42.73, 0.586], ene: [42.7, 0.589], wne: [41.99, 0.577] },
  cnn: { '1he': [45.62, 0.615], w2v: [44.3, 0.605], '2he': [45.12, 0.602], nne: [44.91, 0.603], ene: [43.82, 0.602], wne: [41.59, 0.58] },
  lstm: { '1he': [52.73, 0.646], w2v: [49.13, 0.633], '2he': [47.65, 0.61], nne: [46.8, 0.609], ene: [45.79, 0.601], wne: [43.66, 0.592] },
}

const state = {
  key: 0,
  roman: false,
  seq: [],
  generated: new Set(),
  history: [],
  root: 0,
  model: 'lstm',
  rep: '1he',
  temperature: 1,
}
let vocab, songs, encodings
const nets = {}
const meta = {}
const netKey = (model, rep = state.rep) => `${model}-${rep}`
const modelInfo = id => MODELS.find(m => m.id === id)
const repInfo = id => REPS.find(r => r.id === id)
const cite = ({ section, page }) => `<a href="paper.pdf#page=${page}" target="_blank">§${section}</a>`

// ---------- Chord names ----------

const pretty = suffix => PRETTY[suffix] ?? suffix.replace(/\(\+/g, '(')
const fancy = s => s.replace(/b(?=\d)/g, '♭').replace(/#/g, '♯')
const noteName = pc => (SHARP_KEYS.has(state.key) ? SHARPS : FLATS)[pc % 12]
const chordName = i => noteName(state.key + vocab[i].root) + fancy(pretty(vocab[i].suffix))
const romanName = i => ROMAN[vocab[i].root] + fancy(pretty(vocab[i].suffix))

// Typeset like the paper: the quality on the line, extensions raised (Dm⁷, C^maj7).
function typeset(root, suffix) {
  const [, base, raised] = fancy(pretty(suffix)).match(/^(m(?!aj)|dim|\+)?(.*)$/)
  return `${root}${base ?? ''}${raised ? `<sup>${raised}</sup>` : ''}`
}
const chordHTML = (i, roman = state.roman) => typeset(roman ? ROMAN[vocab[i].root] : noteName(state.key + vocab[i].root), vocab[i].suffix)

// ---------- Reading chords typed by the user ----------

function canon(s) {
  s = s.replace(/♭/g, 'b').replace(/♯/g, '#').replace(/Δ7?/g, 'maj7').replace(/ø7?/g, 'm7b5').replace(/°/g, 'dim')
  s = s.replace(/^M(?!aj)/, 'maj').toLowerCase().replace(/[()\s,]/g, '')
  s = s.replace(/^(-|min|mi(?!n))/, 'm').replace(/ma(?=\d)/, 'maj').replace(/alt/, '#9')
  s = s.replace(/^(\+|aug)(\d+)/, '$2#5').replace(/^aug$/, '+').replace(/sus$/, 'sus4')
  return s
}

const NOTE_PCS = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }
const ROMAN_PCS = { I: 0, II: 2, III: 4, IV: 5, V: 7, VI: 9, VII: 11 }

function parseChord(token) {
  token = token.replace(/♭/g, 'b').replace(/♯/g, '#').replace(/\/[A-G][#b]?$/, '')
  let root, suffix
  const roman = token.match(/^([#b]?)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)(.*)$/)
  const note = token.match(/^([A-Ga-g])([#b]?)(.*)$/)
  if (roman) {
    root = ROMAN_PCS[roman[2].toUpperCase()] + (roman[1] === '#' ? 1 : roman[1] === 'b' ? -1 : 0)
    suffix = roman[3]
    if (roman[2] === roman[2].toLowerCase() && !/^(m|-|dim|ø|°)/.test(suffix)) suffix = 'm' + suffix
  } else if (note) {
    root = NOTE_PCS[note[1].toUpperCase()] + (note[2] === '#' ? 1 : note[2] === 'b' ? -1 : 0) - state.key
    suffix = note[3]
  } else {
    return null
  }
  root = ((root % 12) + 12) % 12
  const wanted = canon(suffix)
  let best = -1, bestScore = -1
  vocab.forEach((v, i) => {
    if (v.root !== root) return
    const have = canon(pretty(v.suffix))
    if (have === wanted) return (best = i), (bestScore = Infinity)
    let common = 0
    while (common < have.length && have[common] === wanted[common]) common++
    const score = common * 1e6 + v.count
    if (score > bestScore) (best = i), (bestScore = score)
  })
  return best < 0 ? null : { index: best, exact: bestScore === Infinity }
}

function readChords(text) {
  const added = [], approx = [], unknown = []
  for (const token of text.split(/[\s|,]+/).filter(Boolean)) {
    const parsed = parseChord(token)
    if (!parsed) unknown.push(token)
    else {
      added.push(parsed.index)
      if (!parsed.exact) approx.push(`${token} → ${chordName(parsed.index)}`)
    }
  }
  const messages = []
  if (approx.length) messages.push(`Not in the dataset, so the closest chord was used: ${approx.join(', ')}`)
  if (unknown.length) messages.push(`Couldn't read ${unknown.join(', ')}`)
  $('message').textContent = messages.join('. ')
  return added
}

function commit(seq, generated = new Set()) {
  state.history.push({ seq: state.seq, generated: state.generated })
  state.seq = seq
  state.generated = generated
  render()
}

// ---------- Models ----------

const bytes = tensors => tensors.reduce((sum, t) => sum + (t.type === 'f32' ? t.shape[0] * 4 : t.shape[0] * (4 + t.shape[1])), 0)
const progress = {}
const loaded = new Set()

async function loadNet(model, rep) {
  const key = netKey(model, rep)
  nets[key] ??= (async () => {
    const expected = bytes(meta[key].tensors)
    const reader = (await fetch(`data/models/${key}.bin`)).body.getReader()
    const chunks = []
    let received = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      received += value.length
      progress[key] = Math.min(1, received / expected)
      showProgress(key)
    }
    const buffer = new Uint8Array(received)
    let offset = 0
    for (const chunk of chunks) buffer.set(chunk, offset), (offset += chunk.length)
    const weights = loadWeights(meta[key].tensors, buffer.buffer)
    loaded.add(key)
    return weights
  })()
  return nets[key]
}

// Probabilities of every chord coming next. Networks read the last few chords through a representation;
// the baselines work on chord names directly.
async function predict(model, seq, rep = state.rep) {
  const size = vocab.length
  if (!seq.length) {
    const total = vocab.reduce((a, v) => a + v.count, 0)
    return vocab.map(v => v.count / total)
  }
  if (model === 'ngram') return ngram(songs.seqs, seq, 8, 10, size).probs
  if (model === 'fifths') return descendingFifth(vocab, seq)
  const weights = await loadNet(model, rep)
  const enc = encodings[rep]
  enc.encode ??= encoder(enc)
  return softmax(networkLogits(model, weights, enc.encode, enc.dim, seq))
}

function withTemperature(probs) {
  const scaled = probs.map(p => p ** (1 / state.temperature))
  const total = scaled.reduce((a, b) => a + b, 0)
  return scaled.map(p => p / total)
}

function sample(probs) {
  let r = Math.random()
  for (let i = 0; i < probs.length; i++) if ((r -= probs[i]) < 0) return i
  return probs.findLastIndex(p => p > 0)
}

const top = (probs, n) => probs.map((p, i) => [i, p]).filter(([, p]) => p > 0).sort((a, b) => b[1] - a[1]).slice(0, n)
const reads = () => state.seq.slice(-WINDOW)

let predictionId = 0
let current = []
const repGuesses = {}

async function updatePredictions() {
  const id = ++predictionId
  renderStrip()
  renderReps()
  renderFigure()
  await new Promise(r => setTimeout(r))
  const probs = await predict(state.model, state.seq)
  if (id !== predictionId) return
  current = withTemperature(probs)
  renderStrip()
  renderFigure()
  renderGuesses()
  renderOthers(id)
  updateRepGuesses(id)
}

// Fills in every representation's guess, downloading the other networks one at a time in the background.
async function updateRepGuesses(id) {
  for (const r of [state.rep, ...REPS.map(r => r.id).filter(r => r !== state.rep)]) {
    const probs = await predict(state.model, state.seq, r)
    if (id !== predictionId) return
    repGuesses[r] = top(probs, 1)[0][0]
    renderReps()
  }
}

async function generate(count) {
  const seq = [...state.seq]
  const generated = new Set(state.generated)
  for (let n = 0; n < count; n++) {
    const probs = n === 0 ? current : withTemperature(await predict(state.model, seq))
    generated.add(seq.length)
    seq.push(sample(probs))
  }
  commit(seq, generated)
  playChords(seq.slice(-Math.min(seq.length, count + 3)))
}

// ---------- Audio ----------

let audio, playback

function audioContext() {
  audio ??= new AudioContext()
  if (audio.state === 'suspended') audio.resume()
  return audio
}

function voicing(i) {
  const { root, comps } = vocab[i]
  const rootPc = (state.key + root) % 12
  const upper = comps.length < 3 ? [0, ...comps] : comps
  return { bass: 36 + rootPc, notes: upper.map(c => 52 + ((((rootPc + c - 52) % 12) + 12) % 12)) }
}

const freq = midi => 440 * 2 ** ((midi - 69) / 12)

function playNote(ctx, out, midi, start, length, { type = 'sine', gain = 0.12, bright = 0.25 } = {}) {
  const env = ctx.createGain()
  env.gain.setValueAtTime(0, start)
  env.gain.linearRampToValueAtTime(gain, start + 0.012)
  env.gain.exponentialRampToValueAtTime(gain * 0.35, start + Math.min(1.2, length))
  env.gain.setTargetAtTime(0, start + length, 0.08)
  env.connect(out)
  for (const [mult, level] of [[1, 1], [2, bright], [3, bright / 3]]) {
    const osc = ctx.createOscillator()
    const g = ctx.createGain()
    osc.type = type
    osc.frequency.value = freq(midi) * mult
    osc.detune.value = (Math.random() - 0.5) * 6
    g.gain.value = level
    osc.connect(g).connect(env)
    osc.start(start)
    osc.stop(start + length + 0.6)
  }
}

function stopPlayback() {
  if (!playback) return
  playback.out.gain.setTargetAtTime(0, audio.currentTime, 0.03)
  playback.timers.forEach(clearTimeout)
  playback = null
  document.querySelectorAll('.chord.playing').forEach(c => c.classList.remove('playing'))
  $('play').textContent = '▶ Play'
}

function playChords(chords, positions) {
  stopPlayback()
  if (!chords.length) return
  const ctx = audioContext()
  const out = ctx.createGain()
  out.gain.value = 0.9
  const filter = ctx.createBiquadFilter()
  filter.type = 'lowpass'
  filter.frequency.value = 3200
  out.connect(filter).connect(ctx.createDynamicsCompressor()).connect(ctx.destination)

  const length = (60 / Number($('tempo').value)) * 2
  const start = ctx.currentTime + 0.05
  const timers = []
  chords.forEach((chord, n) => {
    const at = start + n * length
    const { bass, notes } = voicing(chord)
    playNote(ctx, out, bass, at, length * 0.95, { type: 'triangle', gain: 0.22, bright: 0.1 })
    notes.forEach((m, j) => playNote(ctx, out, m, at + j * 0.012, length * 0.9))
    if (positions) {
      timers.push(setTimeout(() => {
        document.querySelectorAll('.strip .chord').forEach((c, idx) => c.classList.toggle('playing', idx === positions[n]))
      }, (at - ctx.currentTime) * 1000))
    }
  })
  timers.push(setTimeout(() => {
    const looping = positions && $('loop').checked && playback?.out === out
    stopPlayback()
    if (looping) playAll()
  }, (start - ctx.currentTime + chords.length * length) * 1000))
  playback = { out, timers }
  if (positions) $('play').textContent = '■ Stop'
}

const playAll = () => playChords(state.seq, state.seq.map((_, i) => i))

// ---------- Section 1: the progression ----------

function render() {
  $('undo').disabled = !state.history.length
  $('clear').disabled = $('play').disabled = !state.seq.length
  for (const r of REPS) delete repGuesses[r.id]
  renderPalette()
  updatePredictions()
}

function renderStrip() {
  const strip = $('progression')
  const typing = document.activeElement === $('chord-input') ? $('chord-input').value : null
  strip.replaceChildren()
  const firstRead = state.seq.length - reads().length
  state.seq.forEach((chord, pos) => {
    const box = document.createElement('div')
    box.className = 'chord' + (state.generated.has(pos) ? ' generated' : '') + (pos >= firstRead ? ' reading' : '')
    box.title = state.generated.has(pos) ? 'Added by the network' : ''
    box.innerHTML = `<button class="play" aria-label="Play ${chordName(chord)}">${chordHTML(chord)}</button>
      <small>${chordHTML(chord, !state.roman)}</small>
      <button class="remove" aria-label="Remove ${chordName(chord)}">×</button>`
    box.querySelector('.play').onclick = () => playChords([chord])
    box.querySelector('.remove').onclick = () => {
      const generated = new Set([...state.generated].filter(p => p !== pos).map(p => (p > pos ? p - 1 : p)))
      commit(state.seq.filter((_, i) => i !== pos), generated)
    }
    strip.append(box)
  })

  const add = document.createElement('form')
  add.className = 'add'
  add.innerHTML = `<input type="text" id="chord-input" placeholder="${state.seq.length ? 'type a chord…' : 'type chords, e.g. Dm7 G7 Cmaj7'}" autocomplete="off" spellcheck="false" aria-label="Type chords to add">`
  add.onsubmit = e => {
    e.preventDefault()
    const added = readChords($('chord-input').value)
    if (added.length) {
      commit([...state.seq, ...added])
      playChords(state.seq.slice(-4))
      $('chord-input').focus()
    }
  }
  strip.append(add)
  if (typing !== null) {
    $('chord-input').value = typing
    $('chord-input').focus()
  }

  const [best] = top(current, 1)
  if (best) {
    const next = document.createElement('button')
    next.className = 'next'
    next.innerHTML = `${chordHTML(best[0])}<small>add guess?</small>`
    next.title = 'Add the most likely next chord'
    next.onclick = () => {
      commit([...state.seq, best[0]], new Set([...state.generated, state.seq.length]))
      playChords(state.seq.slice(-3))
    }
    strip.append(next)
  }
}

function renderPalette() {
  const roots = $('roots')
  roots.replaceChildren()
  for (let r = 0; r < 12; r++) {
    const b = document.createElement('button')
    b.textContent = state.roman ? ROMAN[r] : noteName(state.key + r)
    b.className = r === state.root ? 'on' : ''
    b.setAttribute('aria-pressed', r === state.root)
    b.onclick = () => ((state.root = r), renderPalette())
    roots.append(b)
  }
  const qualities = $('qualities')
  qualities.replaceChildren()
  for (const q of QUALITIES) {
    const match = parseChord(ROMAN[state.root].replace('♭', 'b').replace('♯', '#') + q)
    if (!match?.exact) continue
    const b = document.createElement('button')
    b.innerHTML = chordHTML(match.index)
    b.onclick = () => {
      commit([...state.seq, match.index])
      playChords([match.index])
    }
    qualities.append(b)
  }
}

// ---------- Section 2: the representations ----------

function renderReps() {
  const chord = state.seq[state.seq.length - 1]
  const net = modelInfo(state.model)
  const best = Math.max(...REPS.map(r => PAPER[state.model][r.id][0]))
  $('reps').innerHTML = REPS.map(r => {
    const [accuracy] = PAPER[state.model][r.id]
    const key = netKey(state.model, r.id)
    const guess = repGuesses[r.id] !== undefined
      ? chordHTML(repGuesses[r.id])
      : nets[key] && !loaded.has(key) ? `<span class="muted small-text" data-progress="${key}">${Math.round((progress[key] ?? 0) * 100)}%</span>` : '<span class="muted">…</span>'
    return `<tr class="${r.id === state.rep ? 'selected' : ''}" data-rep="${r.id}" title="Use ${r.name}">
      <td><b>${r.name}</b>${r.paper ? '*' : ''} <span class="muted">(${cite(r)})</span><div class="rep-desc">${r.desc}</div></td>
      <td>${chord === undefined ? '' : circles(r.id, chord)}</td>
      <td class="num">${accuracy}%<div class="bar"><span style="width:${(accuracy / best) * 100}%"></span></div></td>
      <td class="guess">${guess}</td></tr>`
  }).join('')
  $('reps').querySelectorAll('tr').forEach(tr => (tr.onclick = e => {
    if (e.target.closest('a')) return
    state.rep = tr.dataset.rep
    updatePredictions()
  }))
  $('reps-caption').innerHTML = chord === undefined
    ? 'Figure 1: The six chord representations compared in the paper. * proposed in the paper.'
    : `Figure 1: ${chordName(chord)} in the key of ${noteName(state.key)}, written in each of the six representations, with the ${net.title}'s accuracy in the paper and its guess for your next chord. * proposed in the paper.`
}

function showProgress(key) {
  const el = document.querySelector(`[data-progress="${key}"]`)
  if (el) el.textContent = `${Math.round(progress[key] * 100)}%`
  if (key === netKey(state.model)) $('status').textContent = progress[key] < 1 ? `Downloading the network… ${Math.round(progress[key] * 100)}% of ${(bytes(meta[key].tensors) / 1e6).toFixed(1)} MB` : ''
}

function circles(rep, chord) {
  const { dim, dense, vectors } = encodings[rep]
  if (dense) {
    const v = vectors[chord]
    const max = Math.max(...v.map(Math.abs)) || 1
    return `<div class="circles dense">${v.map(x => `<span class="c" style="background:rgba(0,0,0,${(Math.abs(x) / max).toFixed(2)})"></span>`).join('')}</div>`
  }
  const ones = new Set(vectors[chord])
  // Positions that stand for notes are shaded like the black keys of a piano, as in the paper's figures.
  const isNote = i => rep === 'ene' || rep === 'wne' || ((rep === 'nne' || rep === '2he') && i < 12)
  const cell = i => {
    const key = isNote(i) && BLACK_KEYS.has((state.key + i) % 12) ? ' key' : ''
    const label = (i + 1) % 3 === 0 || (rep === '1he' && ones.has(i)) ? ` data-n="${i + 1}"` : ''
    return `<span class="c${ones.has(i) ? ' on' : key}"${label}>${ones.has(i) ? 1 : 0}</span>`
  }
  let html = ''
  if (rep === '1he') {
    const at = vectors[chord][0]
    const from = Math.max(0, at - 4), to = Math.min(dim, at + 5)
    if (from > 0) html += '<span class="dots">…</span>'
    for (let i = from; i < to; i++) html += cell(i)
    if (to < dim) html += `<span class="dots">… ${dim}</span>`
  } else {
    for (let i = 0; i < dim; i++) {
      if ((rep === 'nne' || rep === '2he') && i === 12) html += '<span class="gap"></span>'
      html += cell(i)
    }
  }
  return `<div class="circles">${html}</div>`
}

// ---------- Section 3: Figure 1.1 and the guesses ----------

function renderFigure() {
  const r = reads()
  const earlier = state.seq.length - r.length
  const [best] = top(current, 1)
  const inputs = r.length
    ? (earlier ? '<span class="fig-chord more">…</span>' : '') + r.map(c => `<span class="fig-chord">${chordHTML(c)}</span>`).join('')
    : '<span class="fig-chord more">none</span>'
  const rep = repInfo(state.rep)
  $('figure').innerHTML = `
    <div class="fig-group"><div class="fig-boxes">${inputs}</div><div class="fig-brace"></div><div class="fig-label">Input Progression</div></div>
    <div class="fig-arrow double"></div>
    <div class="fig-box emphasis"><span class="kicker">Representation</span>${rep.short}${rep.paper ? '*' : ''}</div>
    <div class="fig-arrow"></div>
    <div class="fig-box"><span class="kicker">Model</span>
      <select id="model-select" aria-label="Network">
        ${MODELS.map(m => `<option value="${m.id}" ${m.id === state.model ? 'selected' : ''}>${m.short}</option>`).join('')}
      </select></div>
    <div class="fig-arrow"></div>
    <div class="fig-group"><div class="fig-boxes"><span class="fig-box output">${best ? chordHTML(best[0]) : '?'}</span></div><div class="fig-brace"></div><div class="fig-label">Output Chord</div></div>`
  $('model-select').onchange = e => {
    state.model = e.target.value
    for (const r of REPS) delete repGuesses[r.id]
    updatePredictions()
  }
  const m = modelInfo(state.model)
  $('figure-caption').innerHTML = `Figure 2: The task. The last ${WINDOW} chords are written in ${rep.name} and fed to a ${m.title} (${cite(m)}), which ${m.desc}`
}

function renderGuesses() {
  const body = $('guesses')
  body.replaceChildren()
  const best = top(current, 8)
  const max = best[0]?.[1] ?? 1
  for (const [i, p] of best) {
    const tr = document.createElement('tr')
    tr.title = `Add ${chordName(i)}`
    tr.innerHTML = `<td class="chordname">${chordHTML(i)}</td><td class="muted">${chordHTML(i, !state.roman)}</td>
      <td><div class="bar"><span style="width:${(p / max) * 100}%"></span></div></td>
      <td class="num">${p >= 0.995 ? '100' : p < 0.001 ? '<0.1' : (p * 100).toFixed(1)}% <button class="hear" aria-label="Hear ${chordName(i)} after your chords">▶</button></td>`
    tr.onclick = () => {
      commit([...state.seq, i], new Set([...state.generated, state.seq.length]))
      playChords(state.seq.slice(-3))
    }
    tr.querySelector('.hear').onclick = e => {
      e.stopPropagation()
      playChords([...state.seq.slice(-2), i])
    }
    body.append(tr)
  }
}

async function renderOthers(id) {
  if (!state.seq.length) return ($('others').textContent = '')
  const parts = []
  for (const b of BASELINES) {
    const [[i]] = top(await predict(b.id, state.seq), 1)
    parts.push(`${b.title} (${cite(b)}) <b>${chordHTML(i)}</b>`)
  }
  if (id !== predictionId) return
  $('others').innerHTML = `For comparison, the paper's baselines would say: ${parts.join(', ')}.`
}

// ---------- Section 4: results ----------

function renderResults() {
  const head = `<tr><th>Representation</th>${MODELS.map(m => `<th class="num">${m.short}</th><th class="num">sim.</th>`).join('')}</tr>`
  const body = REPS.map(r => `<tr><td>${r.name}${r.paper ? '*' : ''}</td>${MODELS.map(m => `<td class="num">${PAPER[m.id][r.id][0]}%</td><td class="num muted">${PAPER[m.id][r.id][1]}</td>`).join('')}</tr>`)
  const baselines = [['Random guess', ...PAPER.random], ['N-gram (k = 8)', ...PAPER.ngram], ['Descending fifths', ...PAPER.fifths]]
    .map(([name, a, s], n) => `<tr class="${n === 0 ? 'group' : ''}"><td>${name}</td><td class="num">${a}%</td><td class="num muted">${s}</td><td colspan="${2 * MODELS.length - 2}"></td></tr>`)
  $('results-head').innerHTML = head
  $('results').innerHTML = [...body, ...baselines].join('')
}

// ---------- Setup ----------

async function init() {
  const json = path => fetch(`data/${path}`).then(r => (r.ok ? r.json() : null), () => null)
  ;[vocab, songs, encodings] = await Promise.all(['vocab.json', 'songs.json', 'encodings.json'].map(json))
  await Promise.all(MODELS.flatMap(m => REPS.map(async r => (meta[netKey(m.id, r.id)] = await json(`models/${netKey(m.id, r.id)}.json`)))))

  FLATS.forEach((name, pc) => $('key').add(new Option(SHARP_KEYS.has(pc) ? SHARPS[pc] : name, pc)))
  $('key').onchange = e => ((state.key = Number(e.target.value)), render())
  const setRoman = roman => () => {
    state.roman = roman
    $('show-names').classList.toggle('on', !roman)
    $('show-roman').classList.toggle('on', roman)
    render()
  }
  $('show-names').onclick = setRoman(false)
  $('show-roman').onclick = setRoman(true)

  for (const [name, chords] of Object.entries(PRESETS)) {
    const b = document.createElement('button')
    b.className = 'link'
    b.textContent = name
    b.onclick = () => {
      commit(readChords(chords))
      playAll()
    }
    $('presets').append(b)
  }

  const datalist = $('songs')
  songs.names.forEach((name, i) => datalist.append(new Option(songs.composers[i] ? `${name} (${songs.composers[i]})` : name)))
  $('song').onchange = e => {
    const i = [...datalist.options].findIndex(o => o.value === e.target.value)
    if (i < 0) return
    commit(songs.seqs[i].slice(0, 8))
    $('message').textContent = ''
    e.target.value = ''
    playAll()
  }

  $('play').onclick = () => (playback ? stopPlayback() : playAll())
  $('tempo').oninput = e => ($('tempo-value').textContent = e.target.value)
  $('temperature').oninput = e => ((state.temperature = Number(e.target.value)), updatePredictions())
  $('undo').onclick = () => {
    const prev = state.history.pop()
    if (!prev) return
    state.seq = prev.seq
    state.generated = prev.generated
    render()
  }
  $('clear').onclick = () => commit([])
  $('generate').onclick = () => generate(1)
  $('generate-4').onclick = () => generate(4)

  renderResults()
  state.seq = readChords(PRESETS['ii–V–I'])
  render()
}

init()
