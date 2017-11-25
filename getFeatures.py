import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def getFeatures(songname):
    song, sample_rate = librosa.load(songname, sr=44100)

    # D = np.abs(librosa.stft(song))**2

    # mel = librosa.feature.melspectrogram(S=D)
    # log_mel = librosa.power_to_db(mel, ref=np.max)

    cqt = librosa.cqt(song, sr=sample_rate, fmin=librosa.note_to_hz('C1'),
                        n_bins=84*4, bins_per_octave=12*4, hop_length=64, filter_scale=0.2)
    amp_cqt = librosa.amplitude_to_db(cqt, ref=np.max)

    # s = int(amp_cqt[0].size/2)
    # print(s)
    # print(amp_cqt[:,s])

    plt.figure(figsize=(12,4))

    tempo, beats = librosa.beat.beat_track(y=song, sr=sample_rate, hop_length=512)
    tempo2, quarterBeats = librosa.beat.beat_track(y=song, sr=sample_rate, hop_length=512, bpm=tempo*4)

    quarterBeats = quarterBeats*8

    # librosa.display.specshow(amp_cqt, sr=sample_rate, x_axis='time', y_axis='hz')

    # plt.vlines(librosa.frames_to_time(quarterBeats)/2,
    #         1, 0.5 * sample_rate,
    #         colors='w', linestyles='-', linewidth=2, alpha=0.5)

    # plt.colorbar(format='%+2.0f dB')

    # plt.show()

    avgBeat = np.median(np.diff(beats))
    w = 24 #int(avgBeat / 2)
    features = []
    for q in quarterBeats:
        features.append(amp_cqt[:,q-w:q+w])
    print(str(len(features)) + ' features detected')
    return features, tempo
