import tensorflow
import librosa
import librosa.display
import sys
import matplotlib.pyplot as plt
import numpy as np

def scoreSong(args):
    score(args[0])

def score(songname):
    song, sample_rate = librosa.load("moonlight.mp3", sr=44100)

    mel = librosa.feature.melspectrogram(song, sr=sample_rate, n_mels=128)

    log_mel = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12,4))

    librosa.display.specshow(log_mel, sr=sample_rate, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+02.0f dB')

    plt.show()

if __name__ == "__main__":
    scoreSong(sys.argv)