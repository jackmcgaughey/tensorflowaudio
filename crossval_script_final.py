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

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

v = None
commands = None

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=v, frame_step=v//2)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def load_create_data():
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googeleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
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
    val_size = int(len(list(train_ds.as_numpy_iterator())) * 0.1)
    train_ds = train_ds.shuffle(10)
    val_ds = train_ds.take(val_size)
    train_ds = train_ds.skip(val_size)
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    metrics = history.history
    test_acc = (list(metrics['accuracy'])[int(len(metrics['accuracy'])-1)])
    test_accuracy = test_acc
    return test_accuracy
def kfold(spectrogram_ds, k):
    n = len(spectrogram_ds)
    k_folds = []
    spectrogram_ds = spectrogram_ds.shuffle(10).enumerate()
    for split_point in range(0, n, n // k):
        test_ds = spectrogram_ds.filter(lambda i, data: i >= split_point and i <= split_point + n // k).map(lambda x, y: y)
        train_ds = spectrogram_ds.filter(lambda i, data: i < split_point or i > split_point + n // k).map(lambda x, y: y)
        k_folds.append((train_ds, test_ds))
    return k_folds

def run_crossval(k=10): 
    metrics_dict = defaultdict(list)
    metrics_average_over_v_dict = defaultdict(list)
    param_space = [int(10+245**i) for i in [0, .25, .5, .75, 1, 1.25 ,1.5, 1.58]]
    for v_param in param_space:
        global v
        v = v_param
        print("New v: ", v)
        spectrogram_ds = load_create_data()
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
        num_labels = len(commands)
        for spectrogram, _ in spectrogram_ds.take(1):
            input_shape = spectrogram.shape
        k_folds = kfold(spectrogram_ds, k)
        for train_ds, test_ds in k_folds:
            model = create_model(input_shape, norm_layer, num_labels)
            batch_size=64
            train_ds=train_ds.batch(batch_size)
            test_ds=test_ds.batch(batch_size)
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            train_ds = train_ds.cache().prefetch(AUTOTUNE)
            test_ds = test_ds.cache().prefetch(AUTOTUNE)
            test_accuracy = train_eval_model(model, train_ds, test_ds, epochs=10)
            metrics_dict[v].append(test_accuracy)
        metrics_average_over_v_dict[v].append(sum(list(metrics_dict[v]))/len(list(metrics_dict[v])))
    print(metrics_dict)
    print("=======================================================")
    print(metrics_average_over_v_dict)
    return metrics_dict

if __name__ == "__main__":
    run_crossval(10)