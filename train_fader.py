from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from speech_model import *
import os
from os.path import join, exists
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.backends import cudnn
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from allsstar_dataset import AllsstarLoader
from utils_data import pad_collate, pad_tensor, features_to_vocoder, split_dataset
from torch.utils.data import DataLoader
from StarGAN.model import Generator as GGG
import sys
import copy
import random

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def replace_mask(x, cond, value):
    np.putmask(x, cond, value)
    return x

parser = argparse.ArgumentParser(description='G training script')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('--exp_name', type=str, help='experiment name')
parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
parser.add_argument('--load_ckpt', type=str, default=None, help='load checkpoint location')
parser.add_argument('--save_ckpt', type=str, default=None, help='save checkpoint location')
parser.add_argument('--generate', default=False, action='store_true', help='generate example')
parser.add_argument('--i', type=int, default=0, help='generate for exmaple i')
args = parser.parse_args()

data_folder = "/data/felix/allsstar/copy_world"
mean_path = join(data_folder, 'mean.npy')
std_path = join(data_folder, 'std.npy')
max_path = join(data_folder, 'max.npy')
min_path = join(data_folder, 'min.npy')
allsstar_index_file = "/data/felix/allsstar/data_toy_world/mgc_index.txt"
if args.save_ckpt:
    model_path = join(args.save_ckpt, args.exp_name)
    if not exists(model_path):
        os.mkdir(model_path)

###### GET MEAN AND STD ######
trans = transforms.Compose([
    transforms.Lambda(lambda x: replace_mask(x, x == -10000000000.0, 0)),
])
allsstar_ds = AllsstarLoader(allsstar_index_file, transform=trans)
allsstar_ds_no_trans = allsstar_ds
if not exists(mean_path) or not exists(max_path):
    print("calculating stats")
    max = np.stack([x.max(0) for x,y in allsstar_ds]).max(0)
    min = np.stack([x.min(0) for x,y in allsstar_ds]).min(0)
    mean = np.stack([x.mean(0) for x,y in allsstar_ds]).mean(0)
    std = np.stack([x.std(0) for x,y in allsstar_ds]).std(0)
    print(f"{allsstar_ds.bad_files} bad files!")
    np.save(max_path, max)
    np.save(min_path, min)
    np.save(mean_path, mean)
    np.save(std_path, std)
else:
    print("loading stats")
    max = np.load(max_path)
    min = np.load(min_path)
    mean = np.load(mean_path)
    std = np.load(std_path)

###### PRE-PROCESS DATA ######
trans = transforms.Compose([
    transforms.Lambda(lambda x: replace_mask(x, x == -10000000000.0, 0)),
    #transforms.Lambda(lambda x: (x-mean)/std),  # normalize
    transforms.Lambda(lambda x: (x-min)/(max-min)),  # normalize
    transforms.Lambda(lambda x: np.concatenate((x, np.zeros((x.shape[0],1)).astype(np.float32)), axis=1)),  # add 64th feature (0)
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # convert to tensor
    #transforms.Lambda(lambda x: pad_tensor(x, 256, 0)[:256, :]),  # pad and trim to 256 frames
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # add channel dim
])
allsstar_ds = AllsstarLoader(allsstar_index_file, transform=trans, shuffle=False)
train, val = split_dataset(allsstar_ds)
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=16)#, collate_fn=pad_collate)
val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=16)#, collate_fn=pad_collate)

#G = Generator(c_dim=args.num_classes).to('cuda')
G = NoCompressGenerator32(c_dim=args.num_classes).to('cuda')
print(G)
if args.load_ckpt:
    G.load_state_dict(torch.load(args.load_ckpt))
    print(f"==>loaded checkpoint from {args.load_ckpt}")

if args.opt == 'sgd':
    opt= torch.optim.SGD(G.parameters(), args.lr, momentum=0.9)
elif args.opt == 'adam':
    opt= torch.optim.Adam(G.parameters(), args.lr)
else:
    opt= torch.optim.SGD(G.parameters(), args.lr, momentum=0.9)
sched = ReduceLROnPlateau(opt, 'min', patience=100, verbose=True)

def train():
    G.train()
    eloss = 0.
    for x,c in tqdm(train_loader):
        x,c=x.to('cuda'),c.to('cuda')
        c=label2onehot(c,args.num_classes).to('cuda')
        G.zero_grad()
        x_id=G(x,c)
        loss = torch.mean(torch.abs(x - x_id))
        eloss += loss.item()
        loss.backward()
        opt.step()
    print("train", eloss / len(train_loader))
    return eloss / len(train_loader)

def test():
    G.eval()
    eloss = 0.
    for x,c in tqdm(val_loader):
        x,c=x.to('cuda'),c.to('cuda')
        c=label2onehot(c,args.num_classes).to('cuda')
        x_id=G(x,c)
        loss = torch.mean(torch.abs(x - x_id))
        eloss += loss.item()
    print("val", eloss / len(val_loader))
    return eloss / len(val_loader)

def generate(i):
    # get an example
    x_real, label_org = allsstar_ds[i]
    print(allsstar_ds.class_to_idx)
    label_org = torch.Tensor([label_org])
    x_real, label_org = x_real.unsqueeze(0), label_org[0].view(1)  # select first in batch
    #label_trg = copy.deepcopy(label_org)
    #label_trg[0] = random.randint(0, args.num_classes-1)
    label_trg = 1 - label_org
    print(f"{label_org} => {label_trg}")
    print(f"{allsstar_ds.idx_to_class[int(label_org.item())]} => {allsstar_ds.idx_to_class[int(label_trg.item())]}")
    c_org = label2onehot(label_org, args.num_classes)
    c_trg = label2onehot(label_trg, args.num_classes)
    x_real = x_real.to('cuda')           # Input images.
    c_org = c_org.to('cuda')             # Original domain labels.
    c_trg = c_trg.to('cuda')             # Target domain labels.
    label_org = label_org.to('cuda')     # Labels for computing classification loss.
    label_trg = label_trg.to('cuda')     # Labels for computing classification loss.

    # restore model and run on example
    out1 = G(x_real, c_trg)
    out2 = G(out1, c_org)

    # de-normalize
    out1 = out1[0, 0, :, :63].cpu().detach().numpy()
    out2 = out2[0, 0, :, :63].cpu().detach().numpy()
    x_real = x_real[0, 0, :, :63].cpu().detach().numpy()
    #out1 = (out1 * std) + mean
    #out2 = (out2 * std) + mean
    #x_real = (x_real * std) + mean
    out1 = (out1 * (max-min) + min)
    out2 = (out2 * (max-min) + min)
    x_real = (x_real * (max-min) + min)

    # restore weird numbers with heuristics
    out1[:,60][out1[:,60] < 2] = -10000000000.0
    out2[:,60][out2[:,60] < 2] = -10000000000.0
    x_real[:,60][x_real[:,60] < 2] = -10000000000.0

    out1 = np.clip(out1, x_real.min(), x_real.max())
    out2 = np.clip(out2, x_real.min(), x_real.max())
    features_to_vocoder(x_real, '/home/mlspeech/felixk/tmp/stargan/org')
    features_to_vocoder(out1, '/home/mlspeech/felixk/tmp/stargan/gen1')
    features_to_vocoder(out2, '/home/mlspeech/felixk/tmp/stargan/gen2')
    print("generated")

if args.generate:
    generate(args.i)
    sys.exit(0)

best_val = float("inf")
best_train = float("inf")
writer = SummaryWriter(comment=args.exp_name)
for e in range(1,1000):
    print("epoch", e)
    train_loss = train()
    val_loss = test()
    writer.add_scalar('train', train_loss, e)
    writer.add_scalar('val', val_loss, e)
    sched.step(val_loss)
    if train_loss < best_train:
        best_train = train_loss
        if args.save_ckpt:
            torch.save(G.state_dict(), join(model_path, "train.pt"))
            print("saved train")
    if val_loss < best_val:
        best_val = val_loss
        if args.save_ckpt:
            torch.save(G.state_dict(), join(model_path, "val.pt"))
            print("saved val")

