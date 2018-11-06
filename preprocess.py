import librosa
import numpy as np
import os, sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess
from utils import world_feats, extract_mel_spec, extract_spec
from boltons.fileutils import iter_find_files


def resample(origin_wavpath, target_wavpath, num_workers):
    #wavfiles = [i for i in os.listdir(join(origin_wavpath, spk)) if i.endswith(".wav")]
    os.makedirs(target_wavpath, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wavfiles = iter_find_files(origin_wavpath, "*.wav")
    for wav in wavfiles:
        wav_to = join(target_wavpath, basename(wav))
        wav_from = join(origin_wavpath, wav)
        futures.append(executor.submit(partial(subprocess.call, ['sox', wav_from, "-r", "16000", wav_to])))
        #subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    return 0

def resample_to_16k(origin_wavpath, target_wavpath, num_workers=1):
    os.makedirs(target_wavpath, exist_ok=True)
    spk_folders = os.listdir(origin_wavpath)
    print(f"> Using {num_workers} workers!")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for spk in spk_folders:
        futures.append(executor.submit(partial(resample, spk, origin_wavpath, target_wavpath)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type = int, default = 16000, help = "Sample rate.")
    parser.add_argument("--origin_wavpath", type = str,  help = "The original wav path to resample.")
    parser.add_argument("--target_wavpath", type = str,  help = "The original wav path to resample.")
    parser.add_argument("--features", type = str,  choices=['world', 'mel', 'spect'])
    parser.add_argument("--num_workers", type = int, default = None, help = "The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    origin_wavpath = argv.origin_wavpath
    target_wavpath = argv.target_wavpath
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()

    # The original wav in VCTK is 48K, first we want to resample to 16K
    #resample(origin_wavpath, target_wavpath, num_workers=num_workers)

    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath
    # spk_folders = os.listdir(work_dir)
    # print("processing {} speaker folders".format(len(spk_folders)))
    # print(spk_folders)

    futures = []
    feature_func = {'mel': world_feats, 'mel': extract_mel_spec, 'spect': extract_spec}[argv.features]
    for wav_path in iter_find_files(target_wavpath, "*.wav"):
        #spk_path = os.path.join(work_dir, spk)
        #futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, mc_dir_test, sample_rate)))
        futures.append(executor.submit(partial(feature_func, wav_path, sample_rate)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    sys.exit(0)
