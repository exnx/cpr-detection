import csv
import json
import argparse
import pandas as pd
import os

path = ''
outpath = ''

def read_csv_as_df(path):
    df = pd.read_csv(path)
    return df

def write_json(data_dict, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def main(args):

    df = read_csv_as_df(args.csv)
    video_ids = df['video_id'].tolist()

    # put in dict
    ids_json = {'test': video_ids}

    outpath = os.path.join(args.out_dir, "clip_ids_for_rate_truth.json")
    write_json(ids_json, outpath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', help='input csv file')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')

    args = parser.parse_args()

    main(args)



'''

python csv_to_json.py \
--csv /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv \
--out-dir /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/

'''

