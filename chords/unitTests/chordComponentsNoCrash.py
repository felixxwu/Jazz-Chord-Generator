import random
from lib.chord import Chord
from lib.randomChord import randomChord

# check if random components don't crash
random.seed(420)
for i in range(10000):
    rc = randomChord(random)
    assert isinstance(rc, Chord)
    assert isinstance(rc.toString(), str)
    assert isinstance(rc.toSymbol(), str)