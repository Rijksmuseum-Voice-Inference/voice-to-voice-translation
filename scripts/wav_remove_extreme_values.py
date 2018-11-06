import argparse
from utils_data import wav_remove_extreme_values

parser = argparse.ArgumentParser(description='remove extreme values from wav')
parser.add_argument('--wav', help='path to wav')
args = parser.parse_args()

wav_remove_extreme_values(args.wav)
