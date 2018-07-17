import sys
import numpy as np
from loadmodel import loadmodel
from imageio import imread, imwrite
import scipy
import scipy.io.wavfile
import mido
import librosa
import math
    
sys.path.append("Mask_RCNN/")

from mrcnn import visualize

PIXELSPERINPUT = 509
SECONDSPERINPUT = 6

# midi notes to corresponding piano key 0-127
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

def stft(x, fft_size, hopsamp):
    window = np.hanning(fft_size)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])

def wav_to_spec(fn):
    input_signal, sample_rate = librosa.load(fn, sr=44100)
    stft_mag = np.array([])
    for i in range(len(input_signal)//int(1e6)):
        temp_signal = input_signal[(int(1e6)*i):(int(1e6)*(i+1))]
        fft_size = 4096
        hopsamp = fft_size // 8
        stft_full = stft(temp_signal, fft_size, hopsamp)
        if stft_mag.shape[0] != 0:
            stft_mag = np.concatenate((stft_mag, abs(stft_full)*2))
        else:
            stft_mag = abs(stft_full)*2

    tempo, _ = librosa.beat.beat_track(y=input_signal, sr=sample_rate, hop_length=512)
    
    return stft_mag[:, :512].T, tempo

def analyze_image(image, model, fn):

    print("Executing model on image")
    results = model.detect([image], verbose=1)

    print("Displaying results")
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             MIDINAMES, figsize=(8, 8))
    return to_note_arr_v2(r['rois'], r['class_ids'], r['scores'])

def to_note_arr(bboxes, notes, scores):
    notearr = []
    for i in range(len(bboxes)):
        if scores[i] < 0.7:
            continue
        x1 = bboxes[i][1] * SECONDSPERINPUT
        x2 = bboxes[i][3] * SECONDSPERINPUT
        pos = x1 + ((x2-x1)//2)
        notearr.append((pos/PIXELSPERINPUT, notes[i]))
    return notearr

def to_note_arr_v2(bboxes, notes, scores):
    notearr = []
    for i in range(len(bboxes)):
        if scores[i] < 0.7:
            continue
        x1 = bboxes[i][1] * SECONDSPERINPUT
        x2 = bboxes[i][3] * SECONDSPERINPUT
        pos = (x1 + 1)/PIXELSPERINPUT
        endpos = (x2)/PIXELSPERINPUT
        notearr.append((pos, notes[i], endpos))
    return notearr

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Execute model on a single spectrogram')
    parser.add_argument('-f', '--file', dest='fn', default='alb_esp1.mp3',
                        help='file path for audio file')
    args = parser.parse_args()

    model = loadmodel()

    # load example image, ground truth notes and ground truth mask (just circles drawn in note locations)
    # image = imread(args.image)

    print("CONVERT")
    origImage, tempo = wav_to_spec('ronfaure.mp3')
    print("START")
    notes = []
    ppi = PIXELSPERINPUT
    spi = SECONDSPERINPUT
    win = 1 # window = (win x ppi)
    for i in range(origImage.shape[1]//(ppi//win)):
        image = origImage[:,(ppi*(i//win)):(ppi*((i//win)+1))]
        #image = imread('examples/test6.wav.png')
        image = np.dstack((image, image, image))
        newnotes = analyze_image(image, model, 'examples/test6')
        notes += [(x[0]+((i//win)*spi), x[1], x[2]+((i//win)*spi)) for x in newnotes]

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    microsecondsPerQuarter = int(500000*(tempo/120))
    track.append(mido.MetaMessage('set_tempo', tempo=microsecondsPerQuarter, time=0))
    #track.append(mido.Message('program_change', program=0, time=0))
    timeAdjust = (500000/microsecondsPerQuarter)
    wholenote = 960
    smallestBeat = ((1/16.0)*2)*timeAdjust
    def beatFit(x, base=smallestBeat):
        return base * int(float(x)/base)
    def beatCeil(x, base=smallestBeat):
        return base * math.ceil(float(x)/base)
    for i in range(len(notes)):
        notes[i] = (notes[i][0]*timeAdjust, notes[i][1], notes[i][2]*timeAdjust)
    for i in range(len(notes)):
        if notes[i][2] != -1:
            length = beatCeil(notes[i][2]-notes[i][0])
            endnote = (notes[i][0]+length, notes[i][1]*-1, -1)
            notes.append(endnote)
    notes = sorted(notes, key=lambda x: x[0])
    prevTime = 0
    lastnoteOns = {}
    lastnoteOffs = {}
    skipNextOff = False
    for i in range(len(notes)):
        currentTime = int(notes[i][0]*wholenote)
        print(currentTime)
        if notes[i][1] < 0:
            if lastnoteOffs.get(notes[i][1]*-1):
                lastnoteOffs[notes[i][1]*-1] = currentTime
            if not skipNextOff:
                track.append(mido.Message('note_on', velocity=0, note=notes[i][1]*-1, time=currentTime-prevTime))
            else:
                skipNextOff = False
        else:
            if lastnoteOns.get(notes[i][1]):
                lastnoteOns[notes[i][1]] = currentTime
            if not lastnoteOffs.get(notes[i][1]) or lastnoteOffs.get(notes[i][1]) >= lastnoteOns.get(notes[i][1]):
                track.append(mido.Message('note_on', note=notes[i][1], time=currentTime-prevTime))
            else:
                skipNextOff = True
        prevTime = currentTime
    
    mid.save('test.mid')
    
    print(tempo)

    # from postProcess import postProcessMidi
    # from sheetMusic import sheetMusic

    # chordlist = postProcessMidi(mid, tempo)

    # print(chordlist)

    # sheetMusic('test', chordlist, int(tempo))

