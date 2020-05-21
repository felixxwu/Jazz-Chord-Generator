import xml.etree.ElementTree as ET
from dataset.xmlChildren import getChild, getChildren
from dataset.chordFromXML import ChordXML

def parseFile(file):
    # get the first child element with tag part, this is the song
    root = ET.parse(file).getroot()
    part1 = getChild(root, "part")

    identification = getChild(root, "identification")
    creator = getChild(identification, "creator")
    composer = creator.text

    # get the song key
    attribute = getChild(part1[0], "attributes")
    key = getChild(attribute, "key")
    keynumber = None

    out = {}
    for measure in part1:
        out[measure.attrib["number"]] = {}

        # each measure (beat) has multiple harmonies (chords)
        harmonies = getChildren(measure, "harmony")

        # parse the harmony tag with Chord().toJson and put it into the json file
        chords = []
        for harmony in harmonies:
            chord = ChordXML(harmony, key)
            keynumber = chord.key
            chords += [chord.toJson()]
        # chords = [Chord(harmony, key).toJson() for harmony in harmonies]
        out[measure.attrib["number"]] = chords

    return out, keynumber, composer