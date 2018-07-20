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
def postProcessMidi(notelist, tempo, step):
    notelengths = ['16th','eighth','quarter','half','whole','breve','long','maxima'] #'64th','32nd',

    print(tempo)
    print(step)
    length = notelist[-1][0]+notelist[-1][2]+1
    beats = np.arange(step, length, step)
    print(beats.shape)

    curbeat = 0
    chordlist = []
    chord = []
    #lastNoteEnd = -1
    for i in range(len(notelist)):
        while curbeat < len(beats) and notelist[i][0] > beats[curbeat]:# and notelist[i][0] >= lastNoteEnd:
            chordlist.append(chord)
            chord = []
            curbeat += 1
        length = notelist[i][2]-notelist[i][0]
        if length == 0:
            chordlist.append(chord)
            chord = []
            curbeat += 1
            continue
        duration = 2**int(round(np.log2(length / step)))
        lengthName = notelengths[int(round(np.log2(duration)))]
        chord.append(idxToKey2(notelist[i][1], lengthName, duration))
        # if notelist[i][0] >= lastNoteEnd:
        #     lastNoteEnd = notelist[i][2]-(length/2)

    return chordlist



# Converts the index of a key to its name
#
# key: index of key from [0,84)
# returns a tuple of the form (step, alter, octave, duration, type)
def idxToKey(key, length='64th', duration=1):
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
    return (step, alter, octave, duration, length)

MIDINAMES = np.array(['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',\
            'A0','A#0','B0',\
            'C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1',\
            'C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2',\
            'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',\
            'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',\
            'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5',\
            'C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6',\
            'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',\
            'C8',\
            '-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'])

def idxToKey2(key, length='64th', duration=1):
    note = MIDINAMES[key]
    step = note[0]
    alter = 1 if '#' in note else 0
    octave = note[-1]
    return (step, alter, octave, duration, length)