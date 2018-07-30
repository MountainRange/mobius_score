import sys
import numpy as np
from loadmodel import loadmodel
from imageio import imread, imwrite
import scipy
import scipy.io.wavfile
import mido
import librosa
import math
from audiomisc import ks_key
import argparse
from tqdm import tqdm

    
sys.path.append("Mask_RCNN/")

from mrcnn import visualize

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.55
#config.log_device_placement = True
                                    
sess = tf.Session(config=config)
set_session(sess)

PIXELSPERINPUT = 485
VERTICALCUTOFF = 512
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
    split = int(264600)
    for i in tqdm(range(len(input_signal)//split)):
        temp_signal = input_signal[(split*i):(split*(i+1))]
        fft_size = 16384
        hopsamp = fft_size // 32
        stft_full = stft(temp_signal, fft_size, hopsamp)

        stft_full = abs(stft_full)
        if np.max(stft_full) != 0:
            stft_full = (stft_full - np.mean(stft_full)) / np.std(stft_full)
            stft_full += abs(np.min(stft_full))
            stft_full *= 255.0/np.max(stft_full)
        
        if stft_mag.shape[0] != 0:
            stft_mag = np.concatenate((stft_mag, stft_full))
        else:
            stft_mag = stft_full

    print("Calculating tempo")
    tempo, _ = librosa.beat.beat_track(y=input_signal, sr=sample_rate, hop_length=512)

    print("Calculating music key")
    chroma = librosa.feature.chroma_stft(y=input_signal, sr=sample_rate)
    chroma = [sum(x)/len(x) for x in chroma]
    bestmajor, bestminor = ks_key(chroma)
    if max(bestmajor) > max(bestminor):
        key = np.argmax(bestmajor)
        #         C, Db, D, Eb, E, F, F#, G, Ab, A, Bb, B
        keymap = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5]
    else:
        key = np.argmax(bestminor)
        #         c, c#, d,  eb, e, f, f#, g,  g#, a, bb, b
        keymap = [-3, 4, -1, -6, 1, -4, 3, -2,  5, 0, -5, 2]
    
    return stft_mag[:, :VERTICALCUTOFF].T, tempo, keymap[key]

def analyze_spec(spec, model):
    results = model.detect([spec], verbose=0)

    r = results[0]
    visualize.display_instances(spec, r['rois'], r['masks'], r['class_ids'], 
                                MIDINAMES, figsize=(8, 8))
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

def run_on_spectrogram(spec):

    notes = []
    ppi = PIXELSPERINPUT
    spi = SECONDSPERINPUT
    win = 1 # window = (win x ppi)
    for i in tqdm(range(spec.shape[1]//(ppi//win))):
        spec3c = spec[:,(ppi*(i//win)):(ppi*((i//win)+1))]
        spec3c = np.dstack((spec3c, spec3c, spec3c))
        newnotes = analyze_spec(spec3c, model)
        notes += [(x[0]+((i//win)*spi), x[1], x[2]+((i//win)*spi)) for x in newnotes]

    return notes

def calculate_note_length(notes, shortestBeat, tempo):

    microsecondsPerQuarter = int(500000*(tempo/120))
    timeAdjust = (500000/microsecondsPerQuarter)
    shortestBeat = ((1.0/args.shortest)*2)*timeAdjust
    def beatFit(x, base=shortestBeat):
        return base * round(float(x)/base)
    def beatFloor(x, base=shortestBeat):
        return base * int(float(x)/base)
    def beatCeil(x, base=shortestBeat):
        return base * math.ceil(float(x)/base)
    for i in range(len(notes)):
        notes[i] = (beatFit(notes[i][0]*timeAdjust), notes[i][1], beatFit(notes[i][2]*timeAdjust))
    
    return notes, timeAdjust, shortestBeat

def create_sheets(notes, tempo, shortestBeat, outfn):
    from postProcess import postProcessMidi
    from sheetMusic import sheetMusic

    notes = sorted(notes, key=lambda x: x[0])
    chordlist = postProcessMidi(notes, tempo, shortestBeat)

    sheetMusic(outfn, chordlist, int(tempo), key=key, smallestNote=64)

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

if __name__ == "__main__":

    print("Parsing arguments")
    parser = argparse.ArgumentParser(description='Execute model on a single spectrogram')
    parser.add_argument('-f', '--file', dest='fn', default='input/zitah.mp3',
                        help='file path for audio file')
    parser.add_argument('-s', '--shortest', dest='shortest', default=64,
                        help='shortest note possible')
    parser.add_argument('-o', '--output', dest='out', default='output/out.mid',
                        help='file path for output midi')
    parser.add_argument('-os', '--outsheet', dest='outxml', default='output/out.xml',
                        help='file path for output sheet music')
    args = parser.parse_args()

    print("Loading model")
    model = loadmodel()
    
    print("Converting mp3 to spectrogram")
    spec, tempo, key = wav_to_spec(args.fn)

    print("Analysing spectrogram")
    notes = run_on_spectrogram(spec)

    print("Calculate note start times and end times")
    notes, timeAdjust, shortestBeat = calculate_note_length(notes, args.shortest, tempo)

    print("Convert notes to Musicxml")
    create_sheets(notes, tempo, shortestBeat, args.outxml)
    
    print("Convert notes to Midi")
    create_midi(notes, timeAdjust, args.out)
    
    print(tempo)

