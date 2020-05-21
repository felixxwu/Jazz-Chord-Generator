from chords.symbol.parts.noteSymbol import Note
from chords.chord import Chord
from dataset.xmlChildren import getChild, getChildren

#A:0 B:2 C:3 D:5 E:7 F:8 G:10
class ChordXML:
    def __init__(self, harmony, keytag):
        self.harmony = harmony # music xml harmony tag
        self.key = self.parseKey(keytag) # music xml key tag
        self.root = self.getRoot()
        self.bass = self.parseBass() # using self.harmony
        self.degrees = self.alterDegrees(self.getDegreesFromKind())
        self.degrees.sort()

    def toChord(self):
        return Chord(self.root, self.degrees, self.bass)

    def toJson(self):
        return {
            "root": self.root,
            "bass": self.bass,
            "degrees": self.degrees
        }

    def parseKey(self, keytag):
        fifths = int(getChild(keytag, "fifths").text)
        mode = getChild(keytag, "mode").text

        if (mode == "major"):
            # major c.o.f. starts at C
            return (3 - fifths * 5) % 12
        elif (mode == "minor"):
            # minor starts at A
            return (0 - fifths * 5) % 12

        print("key is not major or minor!??!?!????!!??: " + mode)

    def parseBass(self):
        bassEl = getChild(self.harmony, "bass")
        if (bassEl == None): return None

        bassStepEl = getChild(bassEl, "bass-step")
        bassAlterEl = getChild(bassEl, "bass-alter")
        if (bassAlterEl == None):
            alterBy = 0
        else:
            alterBy = int(bassAlterEl.text)

        bassnote = Note(bassStepEl.text).toNumber()
        return (bassnote + alterBy - self.key - self.root) % 12

    def intervalToInt(self, interval):
        if (interval == 2):  return 2
        if (interval == 3):  return 4
        if (interval == 4):  return 5
        if (interval == 5):  return 7
        if (interval == 6):  return 9    # mostly used in b6 instead of #5
        if (interval == 7):  return 10   # defaults to minor 7
        if (interval == 9):  return 2
        if (interval == 11): return 5
        if (interval == 13): return 9
        print("weird interval?: " + interval)

    def getDegreesFromKind(self):
        kindEl = getChild(self.harmony, "kind")
        if ("text" in kindEl.attrib):
            attr = kindEl.attrib["text"]
            if (attr == ""):       return [4, 7]
            if (attr == "m"):      return [3, 7]
            if (attr == "6"):      return [4, 7, 9]
            if (attr == "7"):      return [4, 7, 10]
            if (attr == "9"):      return [4, 7, 10, 2]
            if (attr == "11"):     return [4, 7, 10, 5]
            if (attr == "13"):     return [4, 7, 10, 9]
            if (attr == "69"):     return [4, 7, 9, 2]
            if (attr == "m6"):     return [3, 7, 9]
            if (attr == "m7"):     return [3, 7, 10]
            if (attr == "m9"):     return [3, 7, 10, 2]
            if (attr == "m11"):    return [3, 7, 10, 5]
            if (attr == "m13"):    return [3, 7, 10, 9]
            if (attr == "m69"):    return [3, 7, 9, 2]
            if (attr == "maj7"):   return [4, 7, 11]
            if (attr == "maj9"):   return [4, 7, 11, 2]
            if (attr == "maj11"):  return [4, 7, 11, 5]
            if (attr == "maj13"):  return [4, 7, 11, 9]
            if (attr == "7alt"):   return [4, 6, 10]
            if (attr == "+"):      return [4, 8]
            if (attr == "sus4"):   return [5, 7]
            if (attr == "7sus4"):  return [5, 7, 10]
            if (attr == "9sus4"):  return [5, 7, 10, 2]
            if (attr == "13sus4"): return [5, 7, 10, 9]
            
        else:
            text = kindEl.text
            if (text == "major"): return [4, 7]
            if (text == "diminished"): return [3, 6]
            if (text == "diminished-seventh"): return [3, 6, 10]
        
        print("weird kind degree!!: " + kindEl.text)

    def alterDegrees(self, degrees):
        degreeEls = getChildren(self.harmony, "degree")
        if (len(degreeEls) == 0): return degrees

        altered = degrees
        for child in degreeEls:
            interval = getChild(child, "degree-value").text
            alter = getChild(child, "degree-alter").text
            dtype = getChild(child, "degree-type").text
            oldValue = self.intervalToInt(int(interval))
            value = oldValue + int(alter)

            if (dtype == "add"):
                if (value not in altered): altered += [value]
            elif (dtype == "subtract"):
                if (value in altered):     altered.remove(value)
            elif (dtype == "alter"):
                if (oldValue in altered):     altered.remove(oldValue)
                if (value not in altered): altered += [value]
            else:
                print("weird degree type!!!: " + dtype)

        return altered

    def getRoot(self) -> int:
        rootEl = getChild(self.harmony, "root")
        rootStepEl = getChild(rootEl, "root-step")
        rootAlterEl = getChild(rootEl, "root-alter")

        note = Note(rootStepEl.text).toNumber()
        if (rootAlterEl == None):
            alter = 0
        else:
            alter = int(rootAlterEl.text)

        return (note + alter - self.key) % 12
    