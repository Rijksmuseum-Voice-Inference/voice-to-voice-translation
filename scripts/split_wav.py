from scipy.io import wavfile
import numpy as np
import argparse
import os
from vad import apply_vad
import traceback


parser = argparse.ArgumentParser(description='Split file according to peaks of left channel, the split wavs are created in a folder with the name of the original wav')
parser.add_argument('--src', help='WAV file to be split', required=True)
parser.add_argument('--vad', help='flag to apply VAD', action='store_true')
parser.add_argument('--vad_threshold', help='threshold for VAD', default=0.05)
parser.add_argument('--vad_filter_size', help='filter size for median smoothing in VAD', default=31)
parser.add_argument('--vad_window', help='"mistake window" size for VAD', default=2000)
parser.add_argument('--threshold', type=int, default=20150, help='split right channel using peaks left channel according to this threshold')
args = parser.parse_args()

print(f"running on '{args.src}'")
print(f"apply vad: {args.vad}")
print(f"left channel threshold: {args.threshold}")

# read wav, quit if only 1 channel
fs, wav = wavfile.read(args.src)
if len(wav.shape) == 1:
    print("no left channel detected! quitting...")
    raise SystemExit

# else, read both channels
right_channel, left_channel = wav[:, 0], wav[:, 1]

# if there are no peaks on left channel -- quit
if left_channel.max() < 100:
    print("no peaks in left channel detected! quitting...")
    raise SystemExit

# if signal is too weak -- quit
if right_channel.max() < 3000:
    print("right channel signal is too weak! quitting...")
    raise SystemExit

# create folder for splits
path_no_ext = args.src[:-4]
name_no_ext = path_no_ext.split('/')[-1]
if not os.path.exists(path_no_ext):
    os.mkdir(path_no_ext)

# perform peak detection
candidates = np.argwhere(left_channel > args.threshold)
boundaries = np.argwhere((candidates[:-1]-candidates[1:]) < -10)
boundaries = candidates[boundaries[:, 0]]
boundaries = np.concatenate([[0], boundaries[:,0], candidates[-1]])
boundaries = boundaries[1:]

# write splits
for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    try:
        name = os.path.join(path_no_ext, name_no_ext + "_" + str(i+1) + ".wav")
        print(name)
        chunk = right_channel[start: end]
        if args.vad:
            chunk = apply_vad(chunk, fs, window=args.vad_window)
        wavfile.write(name, fs, chunk)
    except Exception as e:
        print(f"failed to write split {i}, continuing...")
        traceback.print_exc()
