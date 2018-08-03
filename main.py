import sys
import numpy as np
from imageio import imread, imwrite
import scipy
import scipy.io.wavfile
import argparse

sys.path.append("Mask_RCNN/")

from constants import PIXELSPERINPUT, VERTICALCUTOFF, SECONDSPERINPUT
from audiospec import wav_to_spec
from model import run_model_on_spectrogram
from parse import calculate_note_length, create_sheets, create_midi

# Reduce GPU utilization to 55%, My GPU gets a memory error if it is higher
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.55
#config.log_device_placement = True
                                    
sess = tf.Session(config=config)
set_session(sess)

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

def main(fn, mfn, shortest, out, outxml):
    
    print("Converting mp3 to spectrogram")
    spec, tempo, musicKey = wav_to_spec(fn)

    print("Loading and Running Model")
    notes = run_model_on_spectrogram(spec, mfn)

    print("Calculate note start times and end times")
    notes, timeAdjust, shortestBeat = calculate_note_length(notes, shortest, tempo)

    print("Convert notes to Musicxml")
    create_sheets(notes, tempo, shortestBeat, outxml, musicKey)
    
    print("Convert notes to Midi")
    create_midi(notes, timeAdjust, out)
    
    print(tempo)

def parseArgs():
    print("Parsing arguments")
    parser = argparse.ArgumentParser(description='Execute model on a single spectrogram')
    parser.add_argument('-f', '--file', dest='fn', default='input/zitah.mp3',
                        help='file path for audio file')
    parser.add_argument('-m', '--model', dest='mfn', default='models/mask_rcnn_notes_0122_512_v2.h5',
                        help='file path for model weights')
    parser.add_argument('-s', '--shortest', dest='shortest', default=64,
                        help='shortest note possible')
    parser.add_argument('-o', '--output', dest='out', default='output/out.mid',
                        help='file path for output midi')
    parser.add_argument('-os', '--outsheet', dest='outxml', default='output/out.xml',
                        help='file path for output sheet music')
    args = parser.parse_args()

    main(args.fn, args.mfn, args.shortest, args.out, args.outxml)

if __name__ == "__main__":
    parseArgs()
