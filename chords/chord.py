from chords.symbol.getSymbol import Symbol
from chords.symbol.parts.noteSymbol import Note
from typing import Optional, List

# TODO .toManyHotNotes()
# TODO .toOneHotChords()


class Chord:
    def __init__(self, root, components=None, bass=None):
        if isinstance(root, int) and components != None:
            self.root = root
            self.components = components
            self.bass = bass
        else:
            json = root
            assert "root" in json
            assert "components" in json
            self.root = json["root"]
            self.components = json["components"]
            self.bass = None
            if "bass" in json: self.bass = json["bass"]

        self.components.sort()

    def alterComponents(self, alteration:Optional[dict]):
        if (alteration == None): return None

        if "+" in alteration and alteration["+"] != None:
            for element in alteration["+"]:
                element = element and element % 12
                if not element in self.components:
                    self.components += [element]

        if "-" in alteration and alteration["-"] != None:
            for element in alteration["-"]:
                element = element and element % 12
                if element in self.components:
                    self.components.remove(element)

        self.components.sort()

    def toString(self) -> str:
        bass = ""
        if self.bass != None: bass = f"/{self.bass}"
        return str(self.root) + ": " + str(self.components) + bass

    def toSymbol(self, key=0, includeRoot=True, keyLess=False) -> str:
        return Symbol((self.root + key) % 12, self.components, self.bass).toString(includeRoot, keyLess)

    def getNotes(self, key=0):
        notes = []
        notes += [Note((self.root + key) % 12).toSymbol() + "4"]
        for component in self.components:
            notes += [Note((self.root + component + key) % 12).toSymbol() + "4"]
        return notes

    def getJson(self):
        return {"root": self.root, "components": self.components}
