import sys
import numpy as np
from loadmodel import loadmodel
from imageio import imread
import scipy
import scipy.io.wavfile
import mido
import librosa
    
sys.path.append("Mask_RCNN/")

from mrcnn import visualize

PIXELSPERSECOND = 329

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
    print("Convert to raw")
    input_signal, sample_rate = librosa.load(fn, sr=44100)
    stft_mag = np.array([])
    for i in range(len(input_signal)//int(1e6)):
        temp_signal = input_signal[(int(1e6)*i):(int(1e6)*(i+1))]
        fft_size = 2048
        hopsamp = fft_size // 16
        stft_full = stft(temp_signal, fft_size, hopsamp)
        if stft_mag.shape[0] != 0:
            stft_mag = np.concatenate((stft_mag, abs(stft_full)*2))
        else:
            stft_mag = abs(stft_full)*2
    
    return stft_mag[:, :256].T

def analyze_image(image, model, gt, fn):

    # show ground truth if --ground-truth is enabled
    # if gt:
    #     notes = [x[1] for x in np.load(fn + '_notes.npy')]
    #     mask = np.load(fn + '_mask.npz')['mask']

    #     bboxes = []
    #     for i in range(len(notes)):
    #         notes[i] = int(notes[i])
    #         bboxes.append([0,1,1,1])
    #     notes = np.array(notes)
    #     bboxes = np.array(bboxes)

    #     print("Displaying Ground Truth masks for image")
    #     visualize.display_instances(image, bboxes, mask, notes, 
    #                                 MIDINAMES, figsize=(8, 8))

    print("Executing model on image")
    results = model.detect([image], verbose=1)

    print("Displaying results")
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             MIDINAMES, figsize=(8, 8))
    return to_note_arr(r['rois'], r['class_ids'], r['scores'])

def to_note_arr(bboxes, notes, scores):
    notearr = []
    for i in range(len(bboxes)):
        if scores[i] < 0.5:
            continue
        x1 = bboxes[i][1]
        x2 = bboxes[i][3]
        pos = x1 + ((x2-x1)//2)
        notearr.append((pos/PIXELSPERSECOND, notes[i]))
    return notearr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Execute model on a single spectrogram')
    parser.add_argument('-g', '--ground-truth', dest='gt', action='store_true',
                        help='FOR DEV DEBUGING, NEED MASK FILES: displays ground truth if available')
    parser.add_argument('-f', '--file', dest='fn', default='alb_esp1.mp3',
                        help='file path for audio file')
    args = parser.parse_args()

    model = loadmodel()

    # load example image, ground truth notes and ground truth mask (just circles drawn in note locations)
    # image = imread(args.image)

    print("CONVERT")
    origImage = wav_to_spec('moonlight.mp3')

    print("START")
    notes = []
    pps = PIXELSPERSECOND
    for i in range(origImage.shape[1]//pps):
        image = origImage[:,(pps*i):(pps*(i+1))]
        image = np.dstack((image, image, image))
        newnotes = analyze_image(image, model, args.gt, 'alb_esp1')
        notes += [(x[0]+i, x[1]) for x in newnotes]

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    #track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    #track.append(mido.Message('program_change', program=0, time=0))
    prevTime = 0
    notes = sorted(notes, key=lambda x: x[0])
    for i in range(len(notes)):
        currentTime = int(notes[i][0]*960)
        print(currentTime)
        track.append(mido.Message('note_on', note=notes[i][1], time=currentTime-prevTime))
        prevTime = currentTime
    
    mid.save('test.mid')
    