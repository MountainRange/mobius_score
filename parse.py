
import math
import mido

def calculate_note_length(notes, shortestBeat, tempo):

    microsecondsPerQuarter = int(500000*(tempo/120))
    timeAdjust = (500000/microsecondsPerQuarter)
    shortestBeat = ((1.0/shortestBeat)*2)*timeAdjust
    def beatFit(x, base=shortestBeat):
        return base * round(float(x)/base)
    def beatFloor(x, base=shortestBeat):
        return base * int(float(x)/base)
    def beatCeil(x, base=shortestBeat):
        return base * math.ceil(float(x)/base)
    for i in range(len(notes)):
        notes[i] = (beatFit(notes[i][0]*timeAdjust), notes[i][1], beatFit(notes[i][2]*timeAdjust))
    
    return notes, timeAdjust, shortestBeat

def create_sheets(notes, tempo, shortestBeat, outfn, musicKey):
    from postProcess import postProcessMidi
    from sheetMusic import sheetMusic

    notes = sorted(notes, key=lambda x: x[0])
    chordlist = postProcessMidi(notes, tempo, shortestBeat)

    sheetMusic(outfn, chordlist, int(tempo), key=musicKey, smallestNote=64)

def create_midi(notes, timeAdjust, outfn):

    microsecondsPerQuarter = int(500000/timeAdjust)

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=microsecondsPerQuarter, time=0))
    #track.append(mido.Message('program_change', program=0, time=0))

    for i in range(len(notes)):
        if notes[i][2] != -1:
            length = notes[i][2]-notes[i][0]
            endnote = (notes[i][0]+length, notes[i][1]*-1, -1)
            notes.append(endnote)
    notes = sorted(notes, key=lambda x: x[0])
    prevTime = 0
    wholenote = 960
    for i in range(len(notes)):
        currentTime = int(notes[i][0]*wholenote)
        if notes[i][1] < 0:
            track.append(mido.Message('note_on', velocity=0, note=notes[i][1]*-1, time=currentTime-prevTime))
        else:
            track.append(mido.Message('note_on', note=notes[i][1], time=currentTime-prevTime))
        prevTime = currentTime
    
    mid.save(outfn)