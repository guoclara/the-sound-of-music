import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
from scipy.io.wavfile import write
from skimage import io
from random import shuffle
import os

# number of samples per second (length of wav file = len(y)/sr)
sr = 22050
# number of samples between successive frames (between columns of spectogram)
hop_length = 512
# sample the input with window size 2048 = 1 frame
n_fft = 2048
# partition entire frequency spectrum into 128 evenly spaced frequencies to the human ear (ie mel scale, not absolute)
n_mels = 128

# scale spectogram to pixel
def minmax_imagescaling(S):
    s_min = S.min()
    s_max = S.max()
    S_std = (S - s_min) / (s_max - s_min)
    return (S_std * 255).astype(np.uint8)


# calculate log scaled melspectorgram from wav file
def wav_to_spectogram(filename):
    y, sr = librosa.load(filename)

    S = librosa.feature.melspectrogram(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # log scale
    S = np.log(S + 1e-9)
    return S, S.min(), S.max()


# get image from spectogram and save it as 'name'
def spectogram_img(S, name):
    img = 255 - np.flip(minmax_imagescaling(S), axis=0)
    io.imsave(name, img)
    return img


# plot the spectogram
def visualize_specto(S):
    fig = plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        S, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel"
    )
    plt.show()


# revert from spectogram image array to spectogram
def revert_to_specto(img, s_min, s_max):
    scaled = np.flip((255 - img), axis=0)
    return (scaled / 255) * (s_max - s_min) + s_min


def shuffle_data(inputs, labels):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    shuffled_in = inputs[indices, :, :]
    shuffled_labels = np.take(labels, indices)
    return shuffled_in, shuffled_labels


def main():
    blues = []
    classical = []
    country = []
    disco = []
    hiphop = []
    jazz = []
    metal = []
    pop = []
    reggae = []
    rock = []
    blues_labels = []
    classical_labels = []
    country_labels = []
    disco_labels = []
    hiphop_labels = []
    jazz_labels = []
    metal_labels = []
    pop_labels = []
    reggae_labels = []
    rock_labels = []

    for wav in os.scandir("../data/blues"):
        s, _, _ = wav_to_spectogram(wav.path)
        blues.append(s[:, :1290])
        blues_labels.append(0)
    shuffle(blues)
    blues = np.array(blues)
    # print("blues done")
    for wav in os.scandir("data/classical"):
        s, _, _ = wav_to_spectogram(wav.path)
        classical.append(s[:, :1290])
        classical_labels.append(1)
    shuffle(classical)
    classical = np.array(classical)
    # print("classical done")
    for wav in os.scandir("data/country"):
        s, _, _ = wav_to_spectogram(wav.path)
        country.append(s[:, :1290])
        country_labels.append(2)
    shuffle(country)
    country = np.array(country)
    # print("country done")
    for wav in os.scandir("data/disco"):
        s, _, _ = wav_to_spectogram(wav.path)
        disco.append(s[:, :1290])
        disco_labels.append(3)
    shuffle(disco)
    disco = np.array(disco)
    # print("disco done")
    for wav in os.scandir("data/hiphop"):
        s, _, _ = wav_to_spectogram(wav.path)
        hiphop.append(s[:, :1290])
        hiphop_labels.append(4)
    shuffle(hiphop)
    hiphop = np.array(hiphop)
    # print("hiphop done")
    for wav in os.scandir("data/jazz"):
        if wav.path != "data/jazz/jazz.00054.wav":
            s, _, _ = wav_to_spectogram(wav.path)
            jazz.append(s[:, :1290])
            jazz_labels.append(5)
    shuffle(jazz)
    jazz = np.array(jazz)
    # print("jazz done")
    for wav in os.scandir("data/metal"):
        s, _, _ = wav_to_spectogram(wav.path)
        metal.append(s[:, :1290])
        metal_labels.append(6)
    shuffle(metal)
    metal = np.array(metal)
    # print("metal done")
    for wav in os.scandir("data/pop"):
        s, _, _ = wav_to_spectogram(wav.path)
        pop.append(s[:, :1290])
        pop_labels.append(7)
    shuffle(pop)
    pop = np.array(pop)
    # print("pop done")
    for wav in os.scandir("data/reggae"):
        s, _, _ = wav_to_spectogram(wav.path)
        reggae.append(s[:, :1290])
        reggae_labels.append(8)
    shuffle(reggae)
    reggae = np.array(reggae)
    # print("reggae done")
    for wav in os.scandir("data/rock"):
        s, _, _ = wav_to_spectogram(wav.path)
        rock.append(s[:, :1290])
        rock_labels.append(9)
    shuffle(rock)
    rock = np.array(rock)
    # print("rock done")

    train_data = np.concatenate(
        (
            blues[:80],
            classical[:80],
            country[:80],
            disco[:80],
            hiphop[:80],
            jazz[:80],
            metal[:80],
            pop[:80],
            reggae[:80],
            rock[:80],
        ),
        axis=0,
    )

    train_labels = np.array(
        blues_labels[:80]
        + classical_labels[:80]
        + country_labels[:80]
        + disco_labels[:80]
        + hiphop_labels[:80]
        + jazz_labels[:80]
        + metal_labels[:80]
        + pop_labels[:80]
        + reggae_labels[:80]
        + rock_labels[:80]
    )

    train_data, train_labels = shuffle_data(train_data, train_labels)

    validate_data = np.concatenate(
        (
            blues[80:90],
            classical[80:90],
            country[80:90],
            disco[80:90],
            hiphop[80:90],
            jazz[80:90],
            metal[80:90],
            pop[80:90],
            reggae[80:90],
            rock[80:90],
        ),
        axis=0,
    )

    validate_labels = np.array(
        blues_labels[80:90]
        + classical_labels[80:90]
        + country_labels[80:90]
        + disco_labels[80:90]
        + hiphop_labels[80:90]
        + jazz_labels[80:90]
        + metal_labels[80:90]
        + pop_labels[80:90]
        + reggae_labels[80:90]
        + rock_labels[80:90]
    )

    validate_data, validate_labels = shuffle_data(validate_data, validate_labels)

    test_data = np.concatenate(
        (
            blues[90:],
            classical[90:],
            country[90:],
            disco[90:],
            hiphop[90:],
            jazz[90:],
            metal[90:],
            pop[90:],
            reggae[90:],
            rock[90:],
        ),
        axis=0,
    )

    test_labels = np.array(
        blues_labels[90:]
        + classical_labels[90:]
        + country_labels[90:]
        + disco_labels[90:]
        + hiphop_labels[90:]
        + jazz_labels[90:]
        + metal_labels[90:]
        + pop_labels[90:]
        + reggae_labels[90:]
        + rock_labels[90:]
    )
    #print(test_data.shape)
    # print(test_labels.shape)
    test_data, test_labels = shuffle_data(test_data, test_labels)

    all_data = [(train_data, train_labels), (validate_data, validate_labels), (test_data, test_labels)]
    
    return all_data
    # spectogram, s_min, s_max = wav_to_spectogram("data/pop.00058.wav")
    # print(spectogram.shape)
    # img = spectogram_img(spectogram, "test.png")
    # reverted = revert_to_specto(img, s_min, s_max)
    # wav = librosa.feature.inverse.mel_to_audio(reverted)
    # write('test.wav', sr, wav)
    # playsound('test.wav')


if __name__ == "__main__":
    main()
