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


import numpy as np
def postProcessMidi(mid, tempo):
    step = (60.0 / tempo)
    print(tempo)
    print(step)
    length = mid.length
    print(length)
    beats = np.arange(step, length, step)

    curbeat = 0
    chordlist = []
    chord = []
    currentTime = 0
    for msg in mid:
        currentTime += msg.time
        while curbeat < len(beats) and currentTime > beats[curbeat]:
            chordlist.append(chord)
            chord = []
            curbeat += 1
        if msg.type == 'note_on':
            chord.append(idxToKey(msg.note-21))
    return chordlist



# Converts the index of a key to its name
#
# key: index of key from [0,84)
# returns a tuple of the form (step, alter, octave, duration, type)
def idxToKey(key, length='16th'):
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
    return (step, alter, octave, 1, length)