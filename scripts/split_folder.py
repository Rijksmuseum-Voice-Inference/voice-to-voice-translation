import argparse
import os
from boltons import fileutils
from subprocess import call

parser = argparse.ArgumentParser(description='Script to split all .wav files in a given folder')
parser.add_argument('--index_file', help='Index file created by create_index_file.py', required=True)
args = parser.parse_args()

files = open(args.index_file, 'r').readlines()
for file in files:
    print("--------------------------------")
    file = file.strip()
    file_split = file.split('\t')
    file_path = file_split[0]
    task_lang = file_split[4]
    sub_take = file_split[7]

    if not task_lang == 'ENG':
        print(f"{file_path} -- task language is not ENG, skipping...")
        continue

    if not sub_take == '-1':
        print(sub_take)
        print(f"{file_path} -- not whole-take file, skipping...")
        continue

    call(['python', 'split_wav.py', '--src', file_path, '--vad'])
