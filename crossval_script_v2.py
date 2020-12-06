import os
import pathlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing

# zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
# def get_spec(v):
#     return tf.abs(tf.signal.stft(tf.concat([waveform, zero_padding], 0), frame_length=v, frame_step=v//2))
# spec_range = [get_spec(v) for v in [255 + 35*i for i in range(11)]]

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Try to avoid doing this
v = None
commands = None

## data loading utilities 
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

def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=v, frame_step=v // 2)
    #product to be the same. length and step = 128??
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

# def preprocess_dataset(files):
#     files_ds = tf.data.Dataset.from_tensor_slices(files)
#     output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
#     output_ds = output_ds.map(
#         get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
#     return output_ds

def load_create_data():
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data')
    global commands
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    #from IPython import embed; embed(colors="neutral")
    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return spectrogram_ds


def create_model(input_shape, norm_layer, num_labels):
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

def train_eval_model(model, train_ds, test_ds, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    EPOCHS = epochs

    val_size = int(int(len(train_ds))*0.1)
    train_ds = train_ds.shuffle()
    val_ds = train_ds.take(val_size)
    train_ds = train_ds.skip(val_size)
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    # TODO: need to write the code for running the test and calculating metrics over "test_ds"
    #       --> for now it's probably ok to just return test accuracy, we can talk more about metrics later on
    # Placeholder
    test_accuracy = test_acc
    return test_accuracy

def kfold(spectrogram_ds, k):
    n = len(spectrogram_ds)
    k_folds = []
    spectrogram_ds = spectrogram_ds.shuffle(10).enumerate()
    for split_point in range(0, n, n // k):
        test_ds = spectrogram_ds.filter(lambda i, data: i >= split_point and i <= split_point + n // k)
        train_ds = spectrogram_ds.filter(lambda i, data:i < split_point or i > split_point + n // k)
        k_folds.append((train_ds, test_ds))
    return k_folds


def run_crossval(k=2):
    # Setup stuff for crossval 
    metrics_dict = defaultdict(list)
    param_space = [255 + 35*i for i in range(11)]

    # This lets you run an ipython terminal at this part of the code
    for v_param in param_space:
        # Set global variable
        global v
        v_old = v.copy() if v else "v not set yet"
        v = v_param
        print("Old v: ", v_old)
        print("New v: ", v)
        # Create dataset with chosen param
        spectrogram_ds = load_create_data()
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
        #?
        num_labels = len(commands)
        for spectrogram, _ in spectrogram_ds.take(1):
            input_shape = spectrogram.shape
        from IPython import embed; embed(colors="neutral")
        #input_shape = spectrogram_ds.take(1)[0].shape
        #input_shape is a constant across a value v, so we just need the shape of the spectrogram with v framlength and v//2 framestep
        k_folds = kfold(spectrogram_ds, k)
        for train_ds, test_ds in k_folds:
            model = create_model(input_shape, norm_layer, num_labels)
            # NOTE: This adds the test accuracy returned from train_eval to a dictionary with keys corresponding to the "v" params
            #       metrics_dict look something like this at the end: {255: [0.81, 0.74, 0.55], 290: [0.95, 0.62, 0.87], ...}
            test_accuracy = train_eval_model(model, train_ds, test_ds, epochs=10)
            metrics_dict[v].append(test_accuracy)

    return metrics_dict


if __name__ == "__main__":
    run_crossval(2)