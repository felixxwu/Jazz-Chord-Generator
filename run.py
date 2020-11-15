print("Importing packages...")
from synthesizer import Player, Synthesizer, Waveform

from chords.symbol.getComponents import ChordParts
from chords.ngrams.ngrams import NGrams
from chords.chord import Chord
from dataset.readData import rootAndDegrees as readData
from chords.symbol.parts.noteSymbol import Note
from chords.symbol.getComponents import ChordParts

player = Player()
player.open_stream()
synthesizer = Synthesizer(osc1_waveform=Waveform.triangle, osc1_volume=1, use_osc2=False)


def getKey():
    return Note(input()).toNumber()

def getChordSequence():
    chords = []
    for chordString in input().split():
        chords += [ChordParts(chordString, key).chord]
    return chords

def generateNextChord(key, chords, exclude = []):
    ngrams = NGrams(readData())

    chordsForNgram = list(map(lambda x: x.getJson(), chords))

    print("\nBuilding Ngrams with n = ", len(chordsForNgram) + 1, "...")
    ngrams.build(len(chordsForNgram) + 1)

    while (len(ngrams.getProbs(chordsForNgram)) == 0):
        chordsForNgram = chordsForNgram[1:]
        print("No Ngrams, building with n = ", len(chordsForNgram) + 1, "...")
        ngrams.build(len(chordsForNgram) + 1)

    probs = ngrams.getProbs(chordsForNgram)[:5]
    print("\nMost likely chords:")
    for prob in probs:
        chordName = Chord(prob[0]['root'], prob[0]['components']).toSymbol(key=key)
        print(f"{chordName} - {round(prob[1] * 100)}%")
    print()

    nextChordJson, chance = ngrams.getNext(chordsForNgram, exclude)

    nextChord = Chord(nextChordJson['root'], nextChordJson['components'])

    print("Current chord sequence: ", end="")
    for chord in chords:
        print(chord.toSymbol(key=key), end=" ")
    print()
    print(f"Next chord: {nextChord.toSymbol(key=key)} - {round(chance * 100)}%")

    def getNextChord():
        playChords(chords + [nextChord])
        print("[N]ew chord - [A]dd to sequence - [P]lay again")

        answer = input()
        if (answer == "A" or answer == "a"):
            return nextChord
        elif (answer == "P" or answer == "p"):
            return getNextChord()
        else:
            return generateNextChord(key, chords, exclude + [nextChord])
    
    return getNextChord()

def playChords(chords):
    for chord in chords:
        print("Playing", chord.toSymbol(key=key))
        player.play_wave(synthesizer.generate_chord(chord.getNotes(key=key), 1.0))


print("\nEnter key:")
key = getKey()
print("\nEnter chord sequence:")
chordSequence = getChordSequence()
while True:
    nextChord = generateNextChord(key, chordSequence)
    chordSequence += [nextChord]
