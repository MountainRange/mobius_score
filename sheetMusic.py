

noteNames = ['64th','32nd','16th','eighth','quarter','half','whole']#

# Generates sheet music in xml format from note information
#
# filename: string that the xml file should be named
# notes: array of chord arrays
# tempo: tempo of the song
# key: key signature
# title: string that will appear as the title of the sheet music
# smallestNote: smallest note fraction to be used, (64 = 64th note, 4 = quarter note)
# 
def sheetMusic(filename, notes, tempo=100, key=0, title='test', smallestNote=64, keybeats=3, keytype=4):
    if type(filename) is not str:
        print('filename must be of type str')
        return
    if type(notes) is not list and type(notes[0]) is not list:
        print('notes must be list of lists')
        return
    print('Writing sheet music to ' + filename)
    f=open(filename,"w+")
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n')
    f.write('<score-partwise version="3.0">\n')
    f.write('  <work>\n')
    f.write('    <work-title>' + title + '</work-title>\n')
    f.write('  </work>\n')
    f.write('  <part-list>\n')
    f.write('    <score-part id="P1">\n')
    f.write('      <part-name>Piano</part-name>\n')
    f.write('      <score-instrument id="P1-I3">\n')
    f.write('        <instrument-name>Piano</instrument-name>\n')
    f.write('      </score-instrument>\n')
    f.write('      <midi-instrument id="P1-I3">\n')
    f.write('        <midi-channel>1</midi-channel>\n')
    f.write('        <midi-program>1</midi-program>\n')
    f.write('        <volume>100</volume>\n')
    f.write('        <pan>0</pan>\n')
    f.write('      </midi-instrument>\n')
    f.write('    </score-part>\n')
    f.write('  </part-list>\n')
    f.write('  <part id="P1">\n')

    measurecount = 0
    currentTime = 0
    # write measures to xml
    for i in range(len(notes)):
        measureSize = int(smallestNote * (keybeats/keytype))
        if i % measureSize == 0:
            voices = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
            f.write('    <measure number="' + str(measurecount) + '">\n')
            if measurecount == 0:
                f.write('      <attributes>\n')
                f.write('        <divisions>' + str(int(smallestNote / 4)) + '</divisions>\n')
                f.write('        <key>\n')
                f.write('          <fifths>' + str(key) + '</fifths>\n')
                f.write('        </key>\n')
                f.write('        <time>\n')
                f.write('          <beats>' + str(keybeats) + '</beats>\n')
                f.write('          <beat-type>' + str(keytype) + '</beat-type>\n')
                f.write('        </time>\n')
                f.write('        <staves>2</staves>\n')
                f.write('        <clef number="1">\n')
                f.write('          <sign>G</sign>\n')
                f.write('          <line>2</line>\n')
                f.write('        </clef>\n')
                f.write('        <clef number="2">\n')
                f.write('          <sign>F</sign>\n')
                f.write('          <line>4</line>\n')
                f.write('        </clef>\n')
                f.write('      </attributes>\n')
                f.write('      <direction placement="above">\n')
                f.write('        <direction-type>\n')
                f.write('          <metronome parentheses="yes">\n')
                f.write('            <beat-unit>quarter</beat-unit>\n')
                f.write('            <per-minute>' + str(tempo) + '</per-minute>\n')
                f.write('          </metronome>\n')
                f.write('        </direction-type>\n')
                f.write('      </direction>\n')
            measurecount += 1
        if len(notes[i]) != 0:
            
            duration = notes[i][0][3]

            if currentTime > i:
                f.write('      <backup>\n')
                f.write('        <duration>' + str(currentTime - i) + '</duration>\n')
                f.write('      </backup>\n')
            elif i > currentTime:
                buildRests(f, i - currentTime, smallestNote, 1)

            voice = findVoice(voices, currentTime, duration)
            newChord = True
            # add all notes in chord
            for j in range(len(notes[i])):

                note = notes[i][j]
                if note[3] != duration:
                    newChord = True
                    f.write('      <backup>\n')
                    f.write('        <duration>' + str(duration) + '</duration>\n')
                    f.write('      </backup>\n')
                    duration = note[3]
                    voice = findVoice(voices, currentTime, duration)
                if (currentTime % measureSize) + duration > measureSize:
                    duration = measureSize - (currentTime % measureSize)
                
                f.write('      <note>\n')
                if j != 0 and not newChord:
                    f.write('        <chord/>\n')
                if newChord:
                    newChord = False
                f.write('        <pitch>\n')
                f.write('          <step>' + str(note[0]) + '</step>\n')
                f.write('          <alter>' + str(note[1]) + '</alter>\n')
                f.write('          <octave>' + str(note[2]) + '</octave>\n')
                f.write('        </pitch>\n')
                f.write('        <duration>' + str(duration) + '</duration>\n')
                f.write('        <voice>' + str(voice) + '</voice>\n')
                f.write('        <type>' + str(noteNames[int(np.log2(duration))]) + '</type>\n')
                f.write('        <staff>' + str(1) + '</staff>\n')
                f.write('      </note>\n')
            
            if i + duration > currentTime:
                currentTime = i + duration

        if i % measureSize == measureSize-1:
            if (i + 1) - currentTime > 0:
                buildRests(f, (i + 1) - currentTime, smallestNote, 1)
                currentTime = (i + 1)
            f.write('    </measure>\n')
    if i % measureSize != measureSize-1:
        f.write('    </measure>\n')
    f.write('  </part>\n')
    f.write('</score-partwise>\n')
    f.close()


def findVoice(voices, currentTime, duration):
    for i in range(1,8):
        if currentTime > voices[i]:
            voices[i] = currentTime + duration
            return i
    return 1

import numpy as np
def buildRests(f, restDuration, smallestNote, voice):
    restSize = smallestNote
    while restDuration != 0:
        #print(restDuration)
        numRests = restDuration // restSize
        for i in range(numRests):
            f.write('      <note>\n')
            f.write('        <rest/>\n')
            f.write('        <duration>' + str(restSize) + '</duration>\n')
            f.write('        <voice>' + str(voice) + '</voice>\n')
            f.write('        <type>' + noteNames[int(np.log2(restSize))] + '</type>\n')
            f.write('        <staff>' + str(1) + '</staff>\n')
            f.write('      </note>\n')
            restDuration -= restSize
        restSize //= 2
    return restDuration