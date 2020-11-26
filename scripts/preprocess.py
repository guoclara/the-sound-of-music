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
import tensorflow as tf

# number of samples per second (length of wav file = len(y)/sr)
sr = 22050
# number of samples between successive frames (between columns of spectogram)
hop_length = 512
# sample the input with window size 2048 = 1 frame
n_fft = 2048
# partition entire frequency spectrum into 128 evenly spaced frequencies to the human ear (ie mel scale, not absolute)
n_mels = 128

# scale S values to 0 to factor
def minmax_scaling(S, factor):
    s_min = S.min()
    s_max = S.max()
    S_std = (S - s_min) / (s_max - s_min)
    return (S_std * factor)

# given a spectogram input and model weights, produce the masked input
# according to the model's activations
# if a png or wav file name is given, produce the png/wav file
# if no filter_index is specified, create representations for all filter activations
# png_name and wav_name must be a list of length num_filters in this case

# S: input spectogram
# weights: list of (weight, bias) tuples
def intrepret_activation(S, weights, filter_index = None, png_name = None, wav_name = None):
    filter_weight, bias = weights

    conv = tf.nn.conv2d(S, filter_weight, strides = (1,1), padding = "SAME")
    conv = tf.nn.bias_add(conv, bias) # shape = (batchSz, 128, 128, 32)
    conv = conv.numpy()

    S = np.squeeze(S)

    if filter_index is not None:
        filter_activations = np.reshape(conv[:, :, :, filter_index], (128, 128))
        # scale activations to 0 to 1 to create a mask
        activation_mask = minmax_scaling(filter_activations, 1)
        # element wise multiply to scale spectogram accordingly
        masked = np.multiply(S, activation_mask)
        if png_name is not None:
            masked_img = spectogram_img(masked, png_name)
        if wav_name is not None:
            wav = librosa.feature.inverse.mel_to_audio(masked)
            write(wav_name, sr, wav)
    else:
        masked = []
        # iterate through each filter
        for i in range(tf.shape(conv)[-1]):
            filter_activations = np.reshape(conv[:, :, :, i], (128, 128))
            # scale activations to 0 to 1 to create a mask
            activation_mask = minmax_scaling(filter_activations, 1)
            # element wise multiply to scale spectogram accordingly
            masked_s = np.multiply(S, activation_mask)

            if png_name is not None:
                masked_img = spectogram_img(masked_s, png_name[i])
            if wav_name is not None:
                wav = librosa.feature.inverse.mel_to_audio(masked_s)
                write(wav_name[i], sr, wav)

    return masked

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
    # scale to 0 to 255 (bw png pixel values)
    img = 255 - np.flip(minmax_scaling(S, 255).astype(np.uint8), axis=0)
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

# splits spectogram into 10 square matrices
def make_square(genre, s):
    genre.extend(
        [s[:, :128], 
        s[:, 128:2*128], 
        s[:, 2*128:3*128], 
        s[:, 3*128:4*128], 
        s[:, 4*128:5*128], 
        s[:, 5*128:6*128],
        s[:, 6*128:7*128], 
        s[:, 7*128:8*128], 
        s[:, 8*128:9*128], 
        s[:, 9*128:10*128]]
    )

def play_masked_spectogram_test():
    spectogram, s_min, s_max = wav_to_spectogram("../data/pop/pop.00058.wav")
    spectogram = spectogram[:, 5*128:6*128]
    mask = np.ones((64,64))
    mask = np.pad(mask, 32)
    masked_s = np.multiply(spectogram, mask)
    # img = spectogram_img(spectogram, "test.png")
    # reverted = revert_to_specto(img, s_min, s_max)
    wav = librosa.feature.inverse.mel_to_audio(masked_s)
    write('masked.wav', sr, wav)
    playsound('masked.wav')

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
        make_square(blues, s)
        blues_labels.extend([0]*10)
    shuffle(blues)
    blues = np.array(blues)
    print("blues done")
    for wav in os.scandir("../data/classical"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(classical, s)
        classical_labels.extend([1]*10)
    shuffle(classical)
    classical = np.array(classical)
    print("classical done")
    for wav in os.scandir("../data/country"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(country, s)
        country_labels.extend([2]*10)
    shuffle(country)
    country = np.array(country)
    print("country done")
    for wav in os.scandir("../data/disco"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(disco, s)
        disco_labels.extend([3]*10)
    shuffle(disco)
    disco = np.array(disco)
    print("disco done")
    for wav in os.scandir("../data/hiphop"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(hiphop, s)
        hiphop_labels.extend([4]*10)
    shuffle(hiphop)
    hiphop = np.array(hiphop)
    print("hiphop done")
    for wav in os.scandir("../data/jazz"):
        if wav.path != "../data/jazz/jazz.00054.wav":
            s, _, _ = wav_to_spectogram(wav.path)
            make_square(jazz, s)
            jazz_labels.extend([5]*10)
    shuffle(jazz)
    jazz = np.array(jazz)
    print("jazz done")
    for wav in os.scandir("../data/metal"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(metal, s)
        metal_labels.extend([6]*10)
    shuffle(metal)
    metal = np.array(metal)
    print("metal done")
    for wav in os.scandir("../data/pop"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(pop, s)
        pop_labels.extend([7]*10)
    shuffle(pop)
    pop = np.array(pop)
    print("pop done")
    for wav in os.scandir("../data/reggae"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(reggae, s)
        reggae_labels.extend([8]*10)
    shuffle(reggae)
    reggae = np.array(reggae)
    print("reggae done")
    for wav in os.scandir("../data/rock"):
        s, _, _ = wav_to_spectogram(wav.path)
        make_square(rock, s)
        rock_labels.extend([9]*10)
    shuffle(rock)
    rock = np.array(rock)
    print("rock done")

    train_data = np.concatenate(
        (
            blues[:800],
            classical[:800],
            country[:800],
            disco[:800],
            hiphop[:800],
            jazz[:800],
            metal[:800],
            pop[:800],
            reggae[:800],
            rock[:800],
        ),
        axis=0,
    )

    train_labels = np.array(
        blues_labels[:800]
        + classical_labels[:800]
        + country_labels[:800]
        + disco_labels[:800]
        + hiphop_labels[:800]
        + jazz_labels[:800]
        + metal_labels[:800]
        + pop_labels[:800]
        + reggae_labels[:800]
        + rock_labels[:800]
    )

    train_data, train_labels = shuffle_data(train_data, train_labels)

    validate_data = np.concatenate(
        (
            blues[800:900],
            classical[800:900],
            country[800:900],
            disco[800:900],
            hiphop[800:900],
            jazz[800:900],
            metal[800:900],
            pop[800:900],
            reggae[800:900],
            rock[800:900],
        ),
        axis=0,
    )

    validate_labels = np.array(
        blues_labels[800:900]
        + classical_labels[800:900]
        + country_labels[800:900]
        + disco_labels[800:900]
        + hiphop_labels[800:900]
        + jazz_labels[800:900]
        + metal_labels[800:900]
        + pop_labels[800:900]
        + reggae_labels[800:900]
        + rock_labels[800:900]
    )

    validate_data, validate_labels = shuffle_data(validate_data, validate_labels)

    test_data = np.concatenate(
        (
            blues[900:],
            classical[900:],
            country[900:],
            disco[900:],
            hiphop[900:],
            jazz[900:],
            metal[900:],
            pop[900:],
            reggae[900:],
            rock[900:],
        ),
        axis=0,
    )

    test_labels = np.array(
        blues_labels[900:]
        + classical_labels[900:]
        + country_labels[900:]
        + disco_labels[900:]
        + hiphop_labels[900:]
        + jazz_labels[900:]
        + metal_labels[900:]
        + pop_labels[900:]
        + reggae_labels[900:]
        + rock_labels[900:]
    )
    test_data, test_labels = shuffle_data(test_data, test_labels)
    return (train_data, train_labels, validate_data, validate_labels, test_data, test_labels)


def auralise_test():
    spectogram, s_min, s_max = wav_to_spectogram("../data/pop/pop.00058.wav")
    spectogram = np.reshape(spectogram[:, 5*128:6*128], (-1, 128, 128, 1))
    filter_weight = np.ones((3, 3, 1, 32))
    bias = np.ones(32)
    weights = (filter_weight, bias)
    masked = intrepret_activation(spectogram, weights, filter_index = 0, png_name = "conv1_filter1.png", wav_name = "conv1_filter1.wav")
    playsound('conv1_filter1.wav')
    print(masked)

    # img = spectogram_img(spectogram, "test.png")
    # reverted = revert_to_specto(img, s_min, s_max)
    # wav = librosa.feature.inverse.mel_to_audio(reverted)
    # write('test.wav', sr, wav)

if __name__ == "__main__":
    main()
    # auralise_test()
