import json
import random


SEED = 22
data_path = './videoID_to_class.json'
out_path = './data_split.json'


train_split = 0.7
val_split = 0.15
test_split = 0.15


with open(data_path) as f:

    all_video_ids = list(json.load(f).keys())  # put ids in list

random.Random(SEED).shuffle(all_video_ids)  # shuffles in place
all_count = len(all_video_ids)

train_count = int(all_count * train_split)
val_count = int(all_count * val_split)

val_top_idx = train_count + val_count

# set the splits
train_data = all_video_ids[:train_count]
val_data = all_video_ids[train_count:val_top_idx]
test_data = all_video_ids[val_top_idx:]

# put in json
data_split_json = {}

data_split_json['train'] = train_data
data_split_json['val'] = val_data
data_split_json['test'] = test_data

# serialize
json_object = json.dumps(data_split_json, indent = 4)


with open(out_path, "w") as f:
    f.write(json_object)





