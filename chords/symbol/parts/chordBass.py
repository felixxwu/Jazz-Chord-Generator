from typing import Optional
from chords.symbol.parts.noteSymbol import Note

class Bass:
    def __init__(self, bass:str):
        self.bass:str = bass

    def getBass(self) -> Optional[int]:
        if self.bass == "": return None
        if self.bass[:1] != "/": return None
        
        note = Note(self.bass[1:])
        if not note.isNoteSymbol: return None
        return note.toNumber()