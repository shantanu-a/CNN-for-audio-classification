import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment

AUDIO_DATASET_PATH = "/audio_dataset"



sounds = ['Laughter', 'car_horn', 'dog_barking', 'drilling', 'Fart', 'Guitar', 'Gunshot_and_gunfire', 'Hi-hat', 'Knock',
          'Shatter', 'siren', 'Snare_drum', 'Splash_and_splatter']

class conf:
    sampling_rate = 30000
    duration = 2
    hop_length = 347*duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration


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

def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        input_file = conv_m4a(input_file)
        output_file = os.path.join(output_path, file+'.png')
        create_spectrogram(input_file, output_file)
        print(file)

for sound in sounds:
    input_path = f'{AUDIO_DATASET_PATH}/train/{sound}'
    output_path = f'spectrograms/train/{sound}'
    create_pngs_from_wavs(input_path, output_path)
    print('Finished ', sound)

for sound in sounds:
    input_path = f'{AUDIO_DATASET_PATH}/val/{sound}'
    output_path = f'/spectrograms/val/{sound}'
    create_pngs_from_wavs(input_path, output_path)
    print('Finished ', sound)    

