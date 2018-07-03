import sys
import numpy as np
from loadmodel import loadmodel
from imageio import imread

import argparse

parser = argparse.ArgumentParser(description='Execute model on a single spectrogram')
parser.add_argument('-g', '--ground-truth', dest='gt', action='store_true',
                    help='displays ground truth if available')
parser.add_argument('-i', '--image', dest='image', default='examples/test1.wav.png',
                    help='file path for spectrogram image to analyze, grayscale is assumed')
args = parser.parse_args()

sys.path.append("Mask_RCNN/")

from mrcnn import visualize

model = loadmodel()

# load example image, ground truth notes and ground truth mask (just circles drawn in note locations)
image = imread(args.image)
image = np.dstack((image, image, image))

# midi notes to corresponding piano key 0-127
names = np.array(['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',\
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

# show ground truth if --ground-truth is enabled
if args.gt:
    notes = [x[1] for x in np.load('examples/test1_notes.npy')]
    mask = np.load('examples/test1_mask.npy')

    bboxes = []
    for i in range(len(notes)):
        notes[i] = int(notes[i])
        bboxes.append([0,1,1,1])
    notes = np.array(notes)
    bboxes = np.array(bboxes)

    print("Displaying Ground Truth masks for image")
    visualize.display_instances(image, bboxes, mask, notes, 
                                names, figsize=(8, 8))

print("Executing model on image")
results = model.detect([image], verbose=1)

print("Displaying results")
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            names, figsize=(8, 8))