import json
import random


SEED = 22
data_path = '/vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clips_by_video_id.json'
out_path = '/vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_ids_split_merged.json'


train_split = 0.7
val_split = 0.15
test_split = 0.15


with open(data_path, 'r') as f:
    video_ids_json = json.loads(f.read())

video_ids_list = list(video_ids_json)


random.Random(SEED).shuffle(video_ids_list)  # shuffles in place
all_count = len(video_ids_list)

train_count = int(all_count * train_split)
val_count = int(all_count * val_split)

val_top_idx = train_count + val_count

# set the splits
train_data = video_ids_list[:train_count]
val_data = video_ids_list[train_count:val_top_idx]
test_data = video_ids_list[val_top_idx:]

# put in json
data_split_json = {}



def get_clip_ids(split):

    clips_by_video_id = {}

    for video_id in split:
        annots = video_ids_json[video_id]
        clips_by_video_id[video_id] = []

        for annot_id in annots:
            clips_by_video_id[video_id].append(annot_id)

    return clips_by_video_id


def get_clip_ids_merged(split):

    clips = []

    for video_id in split:
        annots = video_ids_json[video_id]
        # clips[video_id] = []

        for annot_id in annots:
            clips.append(annot_id)

    return clips


data_split_json['train'] = get_clip_ids_merged(train_data)
data_split_json['val'] = get_clip_ids_merged(val_data)
data_split_json['test'] = get_clip_ids_merged(test_data)


# data_split_json['train'] = get_clip_ids(train_data)
# data_split_json['val'] = get_clip_ids(val_data)
# data_split_json['test'] = get_clip_ids(test_data)


# # now we need to loop thru their clips too and put inside dicts
#
# for annot_id, annot
#


# serialize
json_object = json.dumps(data_split_json, indent = 4)


with open(out_path, "w") as f:
    f.write(json_object)


'''

python split_clips.py

'''


