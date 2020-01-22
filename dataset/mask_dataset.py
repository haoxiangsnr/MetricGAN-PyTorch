import os
from torch.utils.data import Dataset
import librosa
import numpy as np
from util.speech_processing import mag_and_phase, compute_PESQ


class MaskDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384, is_train_mode=True):
        """
        Construct mask dataset
        Args:
            dataset (str): the path of dataset list，see "Notes"
            limit (int): the limit of dataset
            offset (int): the offset of dataset
            sample_length(int): the model only support fixed-length input in training, this parameter specify the input size of the model.
            is_train_mode(bool): In training, the model need fixed-length input; in test, the model need fully-length input.
        Notes:
            the format of the waveform dataset is as follows.
            In list file:
            <abs path of noisy wav 1><space><abs path of the clean wav 1>
            <abs path of noisy wav 2><space><abs path of the clean wav 2>
            ...
            <abs path of noisy wav n><space><abs path of the clean wav n>
            e.g.
            In "dataset.txt":
            /home/dog/train/noisy/a.wav /home/dog/train/clean/a.wav
            /home/dog/train/noisy/b.wav /home/dog/train/clean/b.wav
            ...
            /home/dog/train/noisy/x.wav /home/dog/train/clean/x.wav
        Return:
            mixture signals, clean signals, filename
        """
        super().__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.is_train_mode = is_train_mode

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        Returns:
            noisy_mag: 1, F, T
        """
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        noisy, _ = librosa.load(noisy_path, sr=None)
        clean, _ = librosa.load(clean_path, sr=None)

        noisy_mag, noisy_phase, noisy_length = mag_and_phase(noisy)
        clean_mag, clean_phase, clean_length = mag_and_phase(clean)
        assert noisy_length == clean_length

        return noisy, noisy_mag, noisy_phase, clean, clean_mag, name
