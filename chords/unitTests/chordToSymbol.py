from lib.chord import Chord

chord = Chord(3, [3,10], 6)
assert chord.toSymbol() == "Cm7/Eb"
chord = Chord(11, [6,9], None)
assert chord.toSymbol() == "G#dim6sus"
chord = Chord({"root": 5, "components": [2,7,11], "bass": 5})
assert chord.toSymbol() == "DM9sus/D"
