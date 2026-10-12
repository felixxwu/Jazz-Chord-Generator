// Checks that each exported network scores the same in JS as it did in PyTorch after int8 export.
// Run with: node check_web_models.mjs [model-rep ...]
import { readFileSync, readdirSync } from 'node:fs'
import assert from 'node:assert'
import { loadWeights, encoder, networkLogits } from './web/models.js'

const read = f => readFileSync(new URL(`./web/data/${f}`, import.meta.url))
const encodings = JSON.parse(read('encodings.json'))
const { seqs } = JSON.parse(read('songs.json'))
const val = seqs.slice(Math.floor(seqs.length * 0.85))
const argmax = a => a.reduce((best, v, i) => (v > a[best] ? i : best), 0)

const samples = []
val.forEach((song, s) => {
  for (let i = 1 + (s % 7); i < song.length; i += 7) samples.push([song.slice(0, i), song[i]])
})

const names = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync(new URL('./web/data/models', import.meta.url)).filter(f => f.endsWith('.bin')).map(f => f.slice(0, -4))

for (const name of names) {
  const [model, rep] = name.split('-')
  const meta = JSON.parse(read(`models/${name}.json`))
  const buf = read(`models/${name}.bin`)
  const weights = loadWeights(meta.tensors, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length))
  const encode = encoder(encodings[rep])
  const acc = samples.filter(([ctx, y]) => argmax(networkLogits(model, weights, encode, encodings[rep].dim, ctx)) === y).length / samples.length
  console.log(name, 'js', acc.toFixed(4), 'torch', meta.accuracy.toFixed(4))
  assert(Math.abs(acc - meta.accuracy) < 0.03, `${name} differs from its PyTorch accuracy`)
}
