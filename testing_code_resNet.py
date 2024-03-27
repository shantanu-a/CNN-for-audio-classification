# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment


TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/content/submission/testDirectory"
OUTPUT_CSV_ABSOLUTE_PATH = "/content/submission/output.csv"

MAIN_DIRECTORY="submission"
MAIN_DIRECTORY_ABSOLUTE_PATH = os.path.realpath(MAIN_DIRECTORY)

SPECTROGRAM_DIRECTORY_PATH=MAIN_DIRECTORY_ABSOLUTE_PATH+"/spectrograms"


def conv_m4a(input_file):
    with open(input_file, 'rb') as file:
        header = file.read(20)
        if b'\x4D\x34\x41\x20' in header:
            sound = AudioSegment.from_file(input_file, format='m4a')
            out_file = input_file.split('.')[0] + '.wav'
            os.remove(input_file)
            file_handle = sound.export(out_file, format='wav')
            return out_file

        return input_file
    

class conf:
    sampling_rate = 30000
    duration = 2
    hop_length = 347*duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file, sr = conf.sampling_rate)

    if len(y) > conf.samples:
        y = y[0:0+conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    ms = librosa.feature.melspectrogram(y=y, sr=conf.sampling_rate, n_mels=conf.n_mels,
                                        hop_length=conf.hop_length, n_fft=conf.n_fft,
                                        fmin=conf.fmin, fmax=conf.fmax)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=conf.sampling_rate)
    log_ms = log_ms.astype(np.float32)
    fig.savefig(image_file)
    plt.close(fig)


def audio_to_image(file_path):
    input_file = conv_m4a(file_path)
    fileName=file_path.split('/')[-1].split('.')[0]
    output_file = os.path.join(SPECTROGRAM_DIRECTORY_PATH, fileName+'.png')
    create_spectrogram(input_file, output_file)
    return output_file



def predict_class(image_path,model):
    spectrogram = image.load_img(image_path, target_size=(224, 224))
    spectrogram = np.array(spectrogram)
    spec = []
    spec.append(spectrogram)
    spec = np.array(spec)
    return np.argmax(model.predict(spec))
    

def evaluate(file_path):
    weight_file=os.path.join(MAIN_DIRECTORY_ABSOLUTE_PATH, 'resnetfinal.keras')
    model = tf.keras.models.load_model(weight_file)
    image_file_path=audio_to_image(file_path)
    predicted_class=predict_class(image_file_path,model)

    return predicted_class

def test():
    filenames = []
    predictions = []
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction+1)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)

test()