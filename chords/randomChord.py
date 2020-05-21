from lib.chord import Chord

def randomChord(r):
    components = []
    for x in range(r.randint(2, 6)):
        components += [r.randint(1, 11)]
    components = list(set(components))
    components.sort()
    # (Override):
    # components = [3, 11]
    return Chord(r.randint(0, 11), components, None)