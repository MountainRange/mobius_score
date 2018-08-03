#!/usr/bin/env python3

from getFeatures import getFeatures
from postProcess import postProcess
from sheetMusic import sheetMusic
from train import train
import numpy as np
import sys
import tensorflow as tf

def score(songname="moonlightshort.mp3"):

    features, tempo = getFeatures(songname)

    return tensorDemo(features), tempo

def tensorDemo(features):

    clf = train(features)

    #results = tensorEval(eval_data, eval_labels, clf)
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(features).astype("float32")},
      num_epochs=1,
      shuffle=False)

    results = list(clf.predict(input_fn=predict_input_fn))

    result = [p["classes"] for p in results]

    for r in result:
        r[r < 0.3] = 0
        r[r >= 0.3] = 1
        u, c = np.unique(r, return_counts=True)

    out = [list(x) for x in result]
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        song, tempo = score()
    else:
        song, tempo = score(sys.argv[1])
    post = postProcess(song)
    # print('postProcess result:')
    # print(post)
    sheetMusic('test', post, int(tempo))
