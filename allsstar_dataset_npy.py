import os
import os.path
import librosa
import numpy as np
import torch
import torch.utils.data as data
import pyworld as pw
from utils_data import pad_tensor
import random
from collections import defaultdict

AUDIO_EXTENSIONS = ['.npy']

def make_dataset(index_file, max_occurances_per_attribute=4000):
    spects = []
    files = open(index_file, 'r').readlines()
    classes = []
    class_to_idx = {}

    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        file_path = file_split[0]
        speaker_id = file_split[1]
        gender = file_split[2]
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]
        #if orig_lang not in classes: classes.append(orig_lang)
        if speaker_id not in classes: classes.append(speaker_id)

    class_to_idx = {cls: i for i,cls in enumerate(classes)}

    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        file_path = file_split[0]
        speaker_id = file_split[1]
        gender = file_split[2]
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]

        #spects.append((file_path, class_to_idx[orig_lang]))
        spects.append((file_path, class_to_idx[speaker_id]))

    return spects, class_to_idx

class AllsstarLoader(data.Dataset):
    """TODO
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, index_file, transform=None, min_length=256, seq_length=512, shuffle=False):
        spects, class_to_idx = make_dataset(index_file)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + index_file + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.min_length = min_length
        self.seq_length = seq_length
        self.index_file = index_file
        self.spects = spects
        if shuffle:
            random.shuffle(self.spects)
            print("==> shuffled")
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
        self.transform = transform
        self.loader = np.load
        self.bad_files = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        wav_file, target = self.spects[index]
        spect = self.loader(wav_file)

        if spect.shape[0] < self.min_length:
            self.spects.pop(index)
            return self.__getitem__(index)

        if self.transform is not None:
            spect = self.transform(spect)

        return spect, target

    def __len__(self):
        return len(self.spects)
