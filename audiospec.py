
import numpy as np
import librosa
from tqdm import tqdm
from audiomisc import ks_key

from constants import VERTICALCUTOFF, FFT_SIZE, FFT_HOP

def stft(x, fft_size, hopsamp):
    window = np.hanning(fft_size)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])

def wav_to_spec(fn):
    input_signal, sample_rate = librosa.load(fn, sr=44100)
    stft_mag = np.array([])
    split = int(1e6)#int(264600)
    fft_size = FFT_SIZE
    hopsamp = fft_size // FFT_HOP
    for i in tqdm(range(len(input_signal)//split)):
        temp_signal = input_signal[(split*i):(split*(i+1))]
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