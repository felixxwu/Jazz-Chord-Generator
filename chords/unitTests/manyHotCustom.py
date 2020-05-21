from dataset.vectorise import manyHotCustom

for seq in manyHotCustom():
    for chord in seq:
        assert len(chord) == 12
        assert chord[0] >= 0 and chord[0] < 12
        for note in chord[1:]:
            assert note == 1 or note == 0
