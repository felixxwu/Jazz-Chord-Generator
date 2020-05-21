from typing import Union, Optional

class Note:
    def __init__(self, note):
        self.note = note
        self.mapping = {
            "Ab": 11, "A": 0,  "A#": 1,
            "Bb": 1,  "B": 2,  "B#": 3,
            "Cb": 2,  "C": 3,  "C#": 4,
            "Db": 4,  "D": 5,  "D#": 6,
            "Eb": 6,  "E": 7,  "E#": 8,
            "Fb": 7,  "F": 8,  "F#": 9,
            "Gb": 9,  "G": 10, "G#": 11
        }
        self.unique = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#"]

    def isNoteSymbol(self) -> bool:
        for key in self.mapping:
            if self.note.lower() == key.lower(): return True
        return False

    #A:0 B:2 C:3 D:5 E:7 F:8 G:10
    def toNumber(self) -> Optional[int]:
        if self.note in self.mapping:
            return self.mapping[self.note]
        return None

    def toSymbol(self) -> Optional[str]:
        if self.note >= 0 and self.note < 12:
            return self.unique[self.note]
        return None
