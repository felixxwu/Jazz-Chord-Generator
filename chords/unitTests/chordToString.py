from lib.chord import Chord

chord = Chord(2, [4,10], 4)
assert chord.toString() == "2: [4, 10]/4"
chord = Chord(1, [3,10], None)
assert chord.toString() == "1: [3, 10]"
chord = Chord({"root": 3, "components": [4,7], "bass": 4})
assert chord.toString() == "3: [4, 7]/4"
