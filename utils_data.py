import torch
from torch.utils.data.dataset import Subset
import numpy as np
from os.path import exists, join
import os
from scipy.io.wavfile import read, write


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def pad_collate(batch):
    """
    args:
        batch - list of (x_input, labels)
    """
    # find longest sequence
    max_len = max(map(lambda x: x[0].shape[0], batch))

    # pad x_input and x_target according to max_len
    padded_x_input = list(map(lambda x: pad_tensor(x[0], pad=max_len, dim=0), batch))
    padded_x_input = torch.stack(padded_x_input, dim=0)

    labels = list(map(lambda x: x[1], batch))
    labels = torch.LongTensor(labels)

    return padded_x_input, labels

def split_dataset(ds, split=0.1):
    """
    split dataset into 2 datasets according to split ratio
    """
    num_train = len(ds)
    split = int(num_train * split)
    indices = list(range(num_train))

    train_idx, val_idx = indices[split:], indices[:split]
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    return train_ds, val_ds

def array_to_binary_file(data, output_file_name):
    data = np.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()

def wav_remove_extreme_values(path, min_threshold=-30000, max_threshold=30000):
    sr, wav = read(path)
    wav[wav < min_threshold] = 0
    wav[wav > max_threshold] = 0
    write(path, sr, wav)

def features_to_vocoder(x, path):
    assert len(x.shape) == 2
    mgc = x[:, :60]
    lf0 = x[:, 60].reshape(-1,1)
    bap = x[:, 61:63].reshape(-1,1)

    mgc_path = join(path,  'mgc')
    lf0_path = join(path,  'lf0')
    bap_path = join(path,  'bap')

    if not exists(mgc_path):
        os.mkdir(mgc_path)
    if not exists(lf0_path):
        os.mkdir(lf0_path)
    if not exists(bap_path):
        os.mkdir(bap_path)

    array_to_binary_file(mgc, join(mgc_path, 'x.mgc'))
    array_to_binary_file(lf0, join(lf0_path, 'x.lf0'))
    array_to_binary_file(bap, join(bap_path, 'x.bap'))

    os.system(f"echo 'x' > {path}/index.txt")
