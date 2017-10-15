# Post processes the tensorflow result
#
# nodes: array of 88-length arrays of note information
def postProcess(nodes):
    notes = []
    for keyboard in nodes:
        chord = []
        for key in range(88):
            if keyboard[key] == 1:
                chord.append(idxToKey(key))
            elif keyboard[key] != 0:
                print('Unknown key value')
                return
        notes.append(chord)
    return notes

# Converts the index of a key to its name
#
# key: index of key from [0,84)
def idxToKey(key):
    octave = int(key / 12)
    if key % 12 in [1, 3, 6, 8, 10]:
        alter = 1
    else:
        alter = 0
    if key % 12 < 2:
        step = 'C'
    elif key % 12 < 4:
        step = 'D'
    elif key % 12 == 4:
        step = 'E'
    elif key % 12 < 7:
        step = 'F'
    elif key % 12 < 9:
        step = 'G'
    elif key % 12 < 11:
        step = 'A'
    elif key % 12 == 11:
        step = 'B'
    else:
        print('idxToKey failure')
    return (step, alter, octave)