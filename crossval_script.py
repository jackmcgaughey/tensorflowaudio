
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from sklearn.model_selection import KFold

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def load_create_data():
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    # train_files = filenames[:6400]
    # val_files = filenames[6400: 6400 + 800]
    # test_files = filenames[-800:]

    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this 
        # to work in a TensorFlow graph.
        return parts[-2]

    def get_waveform_and_label(file_path):
        label = get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = decode_audio(audio_binary)
        return waveform, label
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    def get_spectrogram(waveform):
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the 
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        #product to be the same. length and step = 128??
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
        return output_ds
    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    num_labels = len(commands)


def create_model():
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32), 
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    return model

def run_crossval():
    data = load_create_data()
    model = create_model()

    # run crossval 
    #train_acc = []
    #test_acc = []
    train_acc = {}
    test_acc = {}
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    def get_spec(v):
        return tf.abs(tf.signal.stft(tf.concat([waveform, zero_padding], 0), frame_length=v, frame_step=v//2))
    spec_range = [get_spec(v) for v in [255 + 35*i for i in range(11)]]
    kf = KFold(n_splits=2)
    for i, indices in enumerate(kf.split(X)):
        train_index = indices[0]
        test_index = indices[1]
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return None 









# import stuff

# load data
# preproc data

# setup model(s)

# run crossval (with same model over different data)
# ---> goal of crossval is to find the "best" (within our range) value of v

# get results and select best hyperparameter
