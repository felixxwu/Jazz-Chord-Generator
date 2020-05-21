from lib.symbol.getComponents import ChordParts

chord = ChordParts("C#min7b5sus2/E", 0).chord
assert chord.toString() == "4: [2, 6, 10]/7"
assert chord.toSymbol() == "C#dim9sus/E"

chord = ChordParts("Gb6/A#", 0).chord
assert chord.toString() == "9: [4, 7, 9]/1"
assert chord.toSymbol() == "F#6/Bb"

chord = ChordParts("BM7", 0).chord
assert chord.toString() == "2: [4, 7, 11]"
assert chord.toSymbol() == "BM7"
