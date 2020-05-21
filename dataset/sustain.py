def sustain(seqs):
    newSeqs = []
    for seq in seqs:
        newSeq = [seq[0]]
        for chord in seq:
            if newSeq[-1] != chord:
                newSeq += [chord]

        newSeqs += [newSeq]
    return newSeqs