from StarGAN.vctk_models import Generator
import librosa
import torch
from utils_data import *
from utils import world_decompose, world_encode_wav, world_speech_synthesis
from torchvision import transforms
import numpy as np
import argparse

trans = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # convert to tensor
    transforms.Lambda(lambda x: x.float()),  # convert to tensor
    transforms.Lambda(lambda x: np.transpose(x)),  # add channel dim
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # add channel dim
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # add batch dim
])

def replace_mask(x, cond, value):
    np.putmask(x, cond, value)
    return x

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def convert(G, wav_file, wav_file_trg, c, device, fs=16000, concat_f0=True):
    # extract features
    f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs, frame_period=5.0, coded_dim=36)

    # transform and normalize
    if concat_f0:
        feats = np.concatenate((f0.reshape(-1,1), coded_sp), axis=1)
    else:
        feats = coded_sp
    mean = feats.mean(0)
    std = feats.std(0)
    coded_sp_norm = (feats - mean) / std
    if concat_f0:
        coded_sp_norm = np.concatenate((coded_sp_norm, np.zeros((coded_sp_norm.shape[0],3)).astype(np.float32)), axis=1)  # add 64th feature (0)
    coded_sp_norm = trans(coded_sp_norm)
    coded_sp_norm = coded_sp_norm.to(device)

    # convert using the generator
    coded_sp_converted_norm = G(coded_sp_norm, c).data.cpu()
    #coded_sp_converted_norm = coded_sp_norm.data.cpu()

    # remove first 2 dims
    coded_sp_converted_norm = coded_sp_converted_norm.squeeze(0).squeeze(0)
    coded_sp_converted_norm = coded_sp_converted_norm.numpy()

    # de-normalize
    coded_sp_converted_norm = np.transpose(coded_sp_converted_norm)
    if concat_f0:
        coded_sp_converted_norm = coded_sp_converted_norm[:, :-3]
    coded_sp_converted_norm = coded_sp_converted_norm * std + mean

    # separate the f0 and sp
    if concat_f0:
        f0, coded_sp_converted = coded_sp_converted_norm[:, 0], coded_sp_converted_norm[:, 1:]
    else:
        f0, coded_sp_converted = f0, coded_sp_converted_norm

    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    f0 = np.ascontiguousarray(f0)
    wav_transformed = world_speech_synthesis(f0=f0, coded_sp=coded_sp_converted, ap=ap, fs=fs, frame_period=5.0)
    librosa.output.write_wav(wav_file_trg, wav_transformed, fs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--g', type=str, help='---')
    parser.add_argument('--c', type=int, help='<`5:---`>')
    parser.add_argument('--c_dim', type=int, help='<`5:---`>')
    parser.add_argument('--src', type=str, help='<`5:---`>')
    parser.add_argument('--trg', type=str, help='<`5:---`>')
    args = parser.parse_args()

    G = Generator(num_speakers=args.c_dim).to('cuda')
    G.load_state_dict(torch.load(args.g, map_location=lambda storage, loc: storage))
    c = torch.zeros(1, args.c_dim)
    c[0,args.c] = 1
    c = c.to('cuda')
    convert(G, args.src, args.trg, c, "cuda", 16000)
