import os
from torch.utils.data import Dataset
import librosa
import torch
import numpy as np
from util.speech_processing import mag_and_phase, compute_PESQ
from torch.nn.utils.rnn import pad_sequence


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

    @staticmethod
    def pad_batch(batch):
        """
        根据 batch 内的最大长度 pad 整个 batch 内的所有样本

        Args:
            batch: [
                (noisy_1, noisy_mag_1, noisy_phase_1, clean_1, clean_mag_1, name_1),
                (noisy_2, noisy_mag_2, noisy_phase_2, clean_2, clean_mag_2, name_2),
                ...
            ]

        Returns:
            noisy_list, clean_list, name_list: [...]
            noisy_mag, noisy_phase, clean_mag: [batch_size, 1, F, Longest_T]
        """
        noisy_list = []
        clean_list = []
        name_list = []
        noisy_phase_list = []

        noisy_mag_list = []
        clean_mag_list = []

        for noisy, noisy_mag, noisy_phase, clean, clean_mag, name in batch:
            noisy_list.append(noisy)
            clean_list.append(clean)
            name_list.append(name)
            noisy_phase_list.append(noisy_phase)

            noisy_mag_list.append(torch.t(torch.tensor(noisy_mag)))  # [batch_size, F, T] => [batch_size, T, F]
            clean_mag_list.append(torch.t(torch.tensor(clean_mag)))

        noisy_mag = pad_sequence(noisy_mag_list, batch_first=True).unsqueeze(1).permute(0, 1, 3, 2)
        clean_mag = pad_sequence(clean_mag_list, batch_first=True).unsqueeze(1).permute(0, 1, 3, 2)

        return noisy_list, clean_list, name_list, noisy_mag, noisy_phase_list, clean_mag

    def __getitem__(self, item):
        """
        Returns:
            noisy_mag: [F, T]
        """
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        noisy, _ = librosa.load(os.path.abspath(os.path.expanduser(noisy_path)), sr=None)
        clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=None)

        noisy_mag, noisy_phase, noisy_length = mag_and_phase(noisy)
        clean_mag, clean_phase, clean_length = mag_and_phase(clean)
        assert noisy_length == clean_length

        return noisy, noisy_mag, noisy_phase, clean, clean_mag, name
