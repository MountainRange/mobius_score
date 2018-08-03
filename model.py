
from tqdm import tqdm
import numpy as np

from constants import PIXELSPERINPUT, SECONDSPERINPUT
from loadmodel import loadmodel

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

def run_model_on_spectrogram(spec, mfn):

    model = loadmodel(mfn)

    print("Running Model")
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

import sys

from mrcnn import visualize
def analyze_spec(spec, model):
    results = model.detect([spec], verbose=0)

    r = results[0]
    # visualize.display_instances(spec, r['rois'], r['masks'], r['class_ids'], 
    #                             MIDINAMES, figsize=(8, 8))
    return to_note_arr_v2(r['rois'], r['class_ids'], r['scores'])