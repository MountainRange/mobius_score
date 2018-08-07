import sys
import numpy as np
from imageio import imread, imwrite
import scipy
import scipy.io.wavfile
import argparse

sys.path.append("Mask_RCNN/")

from constants import PIXELSPERINPUT, VERTICALCUTOFF, SECONDSPERINPUT, MIDINAMES
from audiospec import wav_to_spec
from model import run_model_on_spectrogram
from parse import calculate_note_length, create_sheets, create_midi
from loadmodel import loadmodel

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def init():
    # Reduce GPU utilization to 55%, My GPU gets a memory error if it is higher
    config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.55
    #config.log_device_placement = True
                                        
    sess = tf.Session(config=config)
    set_session(sess)

def setupOnce(mfn):
    init()
    
    print("Loading Model")
    model = loadmodel(mfn)

    return model

def main(fn, mfn, shortest, out, outxml, model=None):

    if not model:
        model = setupOnce(mfn)
    
    print("Converting mp3 to spectrogram")
    spec, tempo, musicKey = wav_to_spec(fn)

    print("Running Model")
    notes = run_model_on_spectrogram(spec, mfn, model)

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
    parser.add_argument('-m', '--model', dest='mfn', default='models/mask_rcnn_notes_0068_1024_v2.h5',
                        help='file path for model weights')
    parser.add_argument('-s', '--shortest', dest='shortest', default=64, type=int
                        help='shortest note possible')
    parser.add_argument('-o', '--output', dest='out', default='output/out.mid',
                        help='file path for output midi')
    parser.add_argument('-os', '--outsheet', dest='outxml', default='output/out.xml',
                        help='file path for output sheet music')
    args = parser.parse_args()

    main(args.fn, args.mfn, args.shortest, args.out, args.outxml)

if __name__ == "__main__":
    parseArgs()
