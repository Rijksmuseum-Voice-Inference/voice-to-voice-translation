import random
import os
from os.path import join, exists
import numpy as np
import torch
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from torchvision import transforms

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from allsstar_dataset_npy import AllsstarLoader
from utils_data import pad_collate, pad_tensor, features_to_vocoder
from torch.utils.data import DataLoader

def str2bool(v):
    return v.lower() in ('true')

def replace_mask(x, cond, value):
    np.putmask(x, cond, value)
    return x

def normalize(x):
    mean = x.mean(0)
    std = x.std(0)
    return (x-mean)/std

def augment(x, flag=False):
    if flag:
        return x + x.data.new(x.size()).normal_(0, 1) * 0.01
    return x

def random_crop(x, size=256):
    n = x.shape[0]
    if n < size:
        return x
    start = random.randint(0, n-size)
    end = start + size
    return x[start:end]

def main(config):
    # For fast training.
    cudnn.benchmark = True

    config.log_dir = join(config.run_dir, config.exp_name)
    config.model_save_dir = join(config.run_dir, config.exp_name,'ckpt')
    config.sample_dir = join(config.run_dir, config.exp_name,'sample')
    config.result_dir = join(config.run_dir, config.exp_name,'result')

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    world_trans = transforms.Compose([
        transforms.Lambda(lambda x: np.concatenate((x, np.zeros((x.shape[0],3)).astype(np.float32)), axis=1)),  # add 64th feature (0)
    ])
    spect_trans = transforms.Compose([
    ])
    mel_spect_trans = transforms.Compose([
    ])

    data_folder = "/".join(config.allsstar_index_file.split("/")[:-1])
    n_features = {'mel': 40, 'spect': 256, 'world': 37}[config.features]
    feat_trans = {'mel': mel_spect_trans, 'spect': spect_trans, 'world': world_trans}[config.features]

    ###### PRE-PROCESS DATA ######

    trans = transforms.Compose([
        #transforms.Lambda(lambda x: x[:, 1:]),  # convert to tensor
        transforms.Lambda(lambda x: random_crop(x, size=config.seq_length)),  # randomly crop X time steps from input
        transforms.Lambda(lambda x: normalize(x)),  # normalize regradless of features
        transforms.Lambda(lambda x: feat_trans(x)),  # use feature-specific transformations
        transforms.Lambda(lambda x: torch.from_numpy(x)),  # convert to tensor
        transforms.Lambda(lambda x: x.float()),  # convert to tensor
        transforms.Lambda(lambda x: pad_tensor(x, config.seq_length, 0)),  # pad and trim to 256 frames
        transforms.Lambda(lambda x: augment(x, config.augment)),  # data augmentaiton
        transforms.Lambda(lambda x: np.transpose(x)),
        transforms.Lambda(lambda x: x.unsqueeze(0)),  # add channel dim
    ])
    allsstar_ds = AllsstarLoader(config.allsstar_index_file, transform=trans, min_length=config.min_length, seq_length=config.seq_length)
    allsstar_loader = DataLoader(allsstar_ds, batch_size=config.batch_size, num_workers=config.num_workers)
    config.c_dim = allsstar_ds.num_classes

    # Solver for training and testing StarGAN.
    solver = Solver(allsstar_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_dis', type=float, default=1, help='weight for discrimination loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--generator_type', help='type of generator (latent representation size)', choices=['G32', 'G16'])
    parser.add_argument('--augment', action="store_true", default=False)

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both', 'allsstar'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--c_do', type=float, default=0.1, help='droupout for C')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=256)
    parser.add_argument('--min_length', type=int, default=200)
    parser.add_argument('--d_subset_features', type=int, default=13)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'generate'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/CelebA_nocrop/images')
    parser.add_argument('--attr_path', type=str, default='data/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')
    parser.add_argument('--asr_save_dir', type=str, default=None)
    ###
    parser.add_argument('--allsstar_index_file', type=str)
    parser.add_argument('--features', type=str, choices=['world','mel', 'spect'])
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--run_dir', type=str)
    ###

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
