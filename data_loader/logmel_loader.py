import pickle
import h5py
import numpy as np

class LogmelLoader:
    def __init__(self, h5_path, wav_name_path):
        self.h5_path = h5_path
        h5_file = h5py.File(h5_path, 'r')
        self.logmelspec = h5_file['logmelspec']
        with open(wav_name_path, 'rb') as f:
            self.wav_names = pickle.load(f)

    def get_embedding(self, file_name):
        index = self.wav_names.index(file_name)
        # convert [2584,64] -> [2584,64,1]
        audio_embedding = np.expand_dims(self.logmelspec[index], axis=-1)
        return audio_embedding


if __name__ == '__main__':
    h5_path = 'data/logspectrogram/log_melspectrogram.h5'
    wav_name_path = 'data/logspectrogram/wav_names.p'
    loader = LogmelLoader(h5_path, wav_name_path)
    wav_name = 'Distorted AM Radio noise.wav'
    embedding = loader.get_embedding(wav_name)
