# Generates sheet music in xml format from note information
#
# filename: string that the xml file should be named
# title: string that will appear as the title of the sheet music
# notes: array of chord arrays
def sheetMusic(filename, notes, tempo=100, key=0):
    if type(filename) is not str:
        print('filename must be of type str')
        return
    if type(notes) is not list and type(notes[0]) is not list:
        print('notes must be list of lists')
        return
    print('Writing sheet music to ' + filename + '.xml')
    f=open(filename + ".xml","w+")
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n')
    f.write('<score-partwise version="3.0">\n')
    f.write('  <work>\n')
    f.write('    <work-title>' + filename + '</work-title>\n')
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

    # write measures to xml
    measurecount = 0
    for i in range(len(notes)):
        if i % 4 == 0:
            f.write('    <measure number="' + str(measurecount) + '">\n')
            if measurecount == 0:
                f.write('      <attributes>\n')
                f.write('        <divisions>4</divisions>\n')
                f.write('        <key>\n')
                f.write('          <fifths>' + str(key) + '</fifths>\n')
                f.write('        </key>\n')
                f.write('        <time>\n')
                f.write('          <beats>4</beats>\n')
                f.write('          <beat-type>4</beat-type>\n')
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
        if len(notes[i]) == 0:
            f.write('      <note>\n')
            f.write('        <rest/>\n')
            f.write('        <duration>4</duration>\n')
            f.write('        <type>quarter</type>\n')
            f.write('      </note>\n')
        else:
            # add all notes in chord
            for j in range(len(notes[i])):
                note = notes[i][j]
                f.write('      <note>\n')
                if j != 0:
                    f.write('        <chord/>\n')
                f.write('        <pitch>\n')
                f.write('          <step>' + str(note[0]) + '</step>\n')
                f.write('          <alter>' + str(note[1]) + '</alter>\n')
                f.write('          <octave>' + str(note[2]) + '</octave>\n')
                f.write('        </pitch>\n')
                f.write('        <duration>' + str(note[3]) + '</duration>\n')
                f.write('        <type>' + str(note[4]) + '</type>\n')
                f.write('      </note>\n')
        if i % 4 == 3:
            f.write('    </measure>\n')
    if i % 4 != 3:
        f.write('    </measure>\n')
    f.write('  </part>\n')
    f.write('</score-partwise>\n')
    f.close()