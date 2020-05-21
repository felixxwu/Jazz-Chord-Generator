import functools
from typing import List, Optional
from chords.symbol.parts.noteSymbol import Note

class Symbol:
    def __init__(self, root:int, components:List[int], bass:Optional[int]):
        self.root = root
        self.components = components
        self.bass = bass

    def toString(self, includeRoot=True, keyLess=False) -> str:
        self.notesLeft = self.components.copy()

        # each method takes notes away the relevant note(s) from notesLeft
        seventh =   self.getSeventh()
        sus =       self.getSus()
        minmaj =    self.getMinMaj()
        augdim =    self.getAugDim()
        additions = self.getAdditions()

        note = ""
        if includeRoot: note = Note(self.root).toSymbol()
        if includeRoot and keyLess: note = str(self.root)
        bass = ""
        if self.bass != None: bass = f"/{Note(self.bass).toSymbol()}"
        return note + minmaj + augdim + seventh + additions + sus + bass

    def getSus(self) -> str:
        r = lambda x: self.notesLeft.remove(x)
        if self.match([ 2, -3, -4,  5]): r(2); r(5); return "sus2/4"
        if self.match([ 2, -3, -4, -5]): r(2);       return "sus2"
        if self.match([-2, -3, -4,  5]): r(5);       return "sus4"
        if self.match([-2, -3, -4, -5]):             return "sus"
        return ""

    def getMinMaj(self) -> str:
        r = lambda x: self.notesLeft.remove(x)
        if self.match([ 3,  4]): r(4); return "" # add(#9) will follow since 3 is still here
        if self.match([-3,  4]): r(4); return ""
        if self.match([ 3, -4]): r(3); return "m"
        if self.match([-3, -4]):       return ""
        return ""

    def getAugDim(self) -> str:
        r = lambda x: self.notesLeft.remove(x)
        if self.match([ 6, -7, -8]): r(6); return "dim"
        if self.match([-6, -7,  8]): r(8); return "aug"
        return ""

    def getSeventh(self) -> str:
        r = lambda x: self.notesLeft.remove(x)
        if self.match([9, 10]): r(9); r(10); return "13"
        if self.match([9, 11]): r(9); r(11); return "M13"
        if self.match([5, 10]): r(5); r(10); return "11"
        if self.match([5, 11]): r(5); r(11); return "M11"
        if self.match([2, 10]): r(2); r(10); return "9"
        if self.match([2, 11]): r(2); r(11); return "M9"

        if self.match([9]): r(9);   return "6"
        if self.match([10]): r(10); return "7"
        if self.match([11]): r(11); return "M7"
        return ""

    def getAdditions(self) -> str:
        additions = []
        if self.match([1]): additions += ["b9"]
        if self.match([2]): additions += ["9"]
        if self.match([3]): additions += ["#9"]
        if self.match([5]): additions += ["11"]
        if self.match([6]): additions += ["#11"]
        if self.match([8]): additions += ["b13"]
        return functools.reduce(lambda prev, curr: f"{prev}(+{curr})", additions, "")

    # returns true if the positive numbers are all in notesLeft AND the negative numbers are not
    def match(self, notes:List[int]) -> bool:
        for note in notes:
            # if positive, note should be in sequence
            if note >= 0 and not note in self.notesLeft:
                return False
            # if negative, note cannot be in sequence
            if note < 0 and -note in self.notesLeft:
                return False

        return True