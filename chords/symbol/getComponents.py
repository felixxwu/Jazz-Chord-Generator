from typing import List, Any, Optional
from chords.symbol.parts.noteSymbol import Note
from chords.symbol.parts.chordQuality import Quality
from chords.symbol.parts.chordExtension import Extension
from chords.symbol.parts.chordAltFifth import Fifth
from chords.symbol.parts.chordSus import Sus
from chords.symbol.parts.chordBass import Bass
from chords.chord import Chord

class ChordParts:
    def __init__(self, input:str, key):
        self.input:str = input
        self.bass:int =                      self.parseChordPart(lambda x: Bass(x).getBass())
        self.sus:Optional[dict] =            self.parseChordPart(lambda x: Sus(x).getAlterations())
        self.altFifth:Optional[dict] =       self.parseChordPart(lambda x: Fifth(x).getAlterations())
        self.extension:Optional[List[int]] = self.parseChordPart(lambda x: Extension(x).getDegrees())
        self.quality:Optional[List[int]] =   self.parseChordPart(lambda x: Quality(x).getDegrees())
        self.root:int =                      self.parseChordPart(lambda x: Note(x).toNumber())

        self.chord = None
        if self.root != None:
            self.chord = Chord((self.root - key) % 12, [], None)
            self.chord.alterComponents({"+": self.quality})
            self.chord.alterComponents({"+": self.extension})
            self.chord.alterComponents(self.altFifth)
            self.chord.alterComponents(self.sus)
            self.chord.bass = self.bass
            self.chord.alterComponents({"-": [self.bass]})

    def parseChordPart(self, lamb) -> Any:
        if len(self.input) == 0: return None
        result, length = self.match(self.input, lamb)
        self.consume(length)
        return result

    def match(self, input:str, function):
        result = function(input)
        # if there is no match
        if result == None and input != "":
            return self.match(input[1:], function)
        else:
            return result, len(input)

    # removes characters from the start of self.input for parsing
    def consume(self, count=1):
        if count > 0:
            self.input = self.input[:-count]
