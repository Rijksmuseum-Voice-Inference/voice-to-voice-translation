import argparse
import os
from boltons import fileutils

parser = argparse.ArgumentParser(description='Create and index file for ALLSSTAR data, index.txt is create in the path of the dataset')
parser.add_argument('--src', help='data dir', required=True)
parser.add_argument('--ext', help='extension to index', required=True)
args = parser.parse_args()

rows = ["path\tspeaker_id\tgender\tnative_language\ttask_language\ttask\ttake\tsub_take\n"]
files = fileutils.iter_find_files(args.src, f"*.{args.ext}")
for file in files:
    sub_take = None
    try:
        file_no_ext = file[:-4].split("/")[-1]
        split = file_no_ext.split("_")
        speaker_id = split[1]
        speaker_gender = split[2]
        native_lang = split[3]
        task_lang = split[4]
        task = split[5]
        take = split[6]
        sub_take = split[7]
        rows.append(f"{file}\t{speaker_id}\t{speaker_gender}\t{native_lang}\t{task_lang}\t{task}\t{take}\t{sub_take}\n")
    except Exception:
        if sub_take == None:
            sub_take = -1
            rows.append(f"{file}\t{speaker_id}\t{speaker_gender}\t{native_lang}\t{task_lang}\t{task}\t{take}\t{sub_take}\n")
        else:
            print(f"warning: '{file}' has an invalid naming convention (skipped)")

index_file = os.path.join(args.src, f"{args.ext}_index.txt")
open(index_file, 'w').writelines(rows)
print("done")
