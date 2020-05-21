from dataset.vectorise import manyHotNotes

for seq in manyHotNotes():
    for chord in seq:
        assert len(chord) == 24
        for note in chord:
            assert note == 1 or note == 0
