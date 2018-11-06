from tqdm import tqdm
import os
from shutil import copyfile
import os.path as path
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--index_file', type=str, help='---')
parser.add_argument('--target', type=str, help='---')
parser.add_argument('--att', type=str, help='<`5:---`>')
args = parser.parse_args()

def split(index_file, target_folder, att):
    files = open(index_file, 'r').readlines()
    files_by_att = {}

    for file in files[1:]:
        file = file.strip()
        file_split = file.split('\t')
        file_path = file_split[0]
        gender = file_split[2]
        orig_lang = file_split[3]
        task_lang = file_split[4]
        sub_take = file_split[7]

        if sub_take == '-1':
            continue

        if orig_lang not in files_by_att:
            files_by_att[orig_lang] = [file_path]
        else:
            files_by_att[orig_lang].append(file_path)

    os.makedirs(target_folder, exist_ok=True)
    for att in tqdm(files_by_att.keys()):
        att_folder = path.join(target_folder, att)
        os.makedirs(att_folder, exist_ok=True)
        for f in files_by_att[att]:
            copyfile(f, path.join(att_folder, att+"_"+path.basename(f)))

if __name__ == '__main__':
    split(args.index_file, args.target, args.att)



