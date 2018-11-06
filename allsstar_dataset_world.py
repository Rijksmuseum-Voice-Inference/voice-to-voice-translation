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

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = int(features.size / dimension)
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return  features, frame_number

def find_classes(index_file):
    classes = set()
    files = open(index_file, 'r').readlines()
    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]

        if not task_lang == 'ENG':
            continue
        if sub_take == '-1':
            continue

        classes.add(orig_lang)

    classes = list(classes)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(index_file, max_occurances_per_attribute=4000, features="world"):
    spects = []

    allowed_labels = ["ENG", "CMN"]
    class_to_idx = {"M": 0, "F": 1}
    attribute_counter = defaultdict(int)
    files = open(index_file, 'r').readlines()

    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        file_path = file_split[0]
        gender = file_split[2]
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]

        f0_file = file_path.replace("/mgc/", "/lf0/").replace(".mgc", ".lf0")
        bap_file = file_path.replace("/mgc/", "/bap/").replace(".mgc", ".bap")

        if not task_lang == 'ENG':
            continue
        if sub_take == '-1':
            continue
        if not orig_lang in allowed_labels:
           continue
        if attribute_counter[orig_lang] >= max_occurances_per_attribute:
           continue

        attribute_counter[orig_lang] += 1
        if features == "world":
            spects.append((file_path, f0_file, bap_file, class_to_idx[gender]))
        elif features == "mel":
            spects.append((file_path, class_to_idx[gender]))
        #spects.append((file_path, f0_file, bap_file, allowed_labels.index(orig_lang)))

    #class_to_idx = {label:allowed_labels.index(label) for label in allowed_labels}
    return spects, class_to_idx

def make_dataset1(index_file, max_occurances_per_attribute=4000):
    spects = []

    allowed_labels = ["ENG", "CMN"]
    class_to_idx = {"M": 0, "F": 1}
    attribute_counter = defaultdict(int)
    files = open(index_file, 'r').readlines()

    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        file_path = file_split[0]
        gender = file_split[2]
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]

        f0_file = file_path.replace("/mgc/", "/lf0/").replace(".mgc", ".lf0")
        bap_file = file_path.replace("/mgc/", "/bap/").replace(".mgc", ".bap")

        if not task_lang == 'ENG':
            continue
        if sub_take == '-1':
            continue
        if not orig_lang in allowed_labels:
           continue
        if attribute_counter[orig_lang] >= max_occurances_per_attribute:
           continue

        attribute_counter[orig_lang] += 1
        spects.append((file_path, f0_file, bap_file, class_to_idx[gender]))
        #spects.append((file_path, f0_file, bap_file, allowed_labels.index(orig_lang)))

    #class_to_idx = {label:allowed_labels.index(label) for label in allowed_labels}
    return spects, class_to_idx

def world_loader(mgc_file, f0_file, bap_file):
    mgc, mgc_len = load_binary_file_frame(mgc_file, 60)
    f0, f0_len = load_binary_file_frame(f0_file, 1)
    bap, bap_len = load_binary_file_frame(bap_file, 1)
    assert mgc_len == f0_len and mgc_len * 2 == bap_len
    bap = bap.reshape(mgc_len, -1)
    features = np.concatenate((mgc, f0, bap), axis=1)

    return features

def mel_log_spec_loader(path, normalize=True):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = 512
    win_length = 512
    hop_length = 256

    # STFT
    D = librosa.feature.melspectrogram(y, sr=sr, n_mels=40)
    D = np.log1p(D)

    # z-score normalization
    if normalize:
        mean = D.mean()
        D -= mean

    D = np.transpose(D)

    return D

def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


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

    def __init__(self, index_file, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, shuffle=False, features="world"):
        spects, class_to_idx = make_dataset(index_file, features=features)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + index_file + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.index_file = index_file
        self.spects = spects
        if shuffle:
            random.shuffle(self.spects)
            print("shuffled")
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.features = features
        if features == "world":
            self.loader = world_loader
        elif features == "mel":
            self.loader = mel_log_spec_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.bad_files = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        if self.features == "world":
            mgc_file, f0_file, bap_file, target = self.spects[index]
            try:
                spect = self.loader(mgc_file, f0_file, bap_file)
            except Exception as e:
                print(e)
                self.bad_files += 1
                self.spects.pop(index)  # remove bad file
                return self.__getitem__(index)  # return next file instead (technically, this is still the "same" index)

        elif self.features == "mel":
            wav_file, target = self.spects[index]
            spect = self.loader(wav_file)

        if self.transform is not None:
            spect = self.transform(spect)

        return spect, target

    def __len__(self):
        return len(self.spects)
