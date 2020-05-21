from dataset.vectorise import oneHotChords

ohc, unique = oneHotChords()
for seq in ohc:
    for chord in seq:
        assert chord >= 0
        assert chord < len(unique)