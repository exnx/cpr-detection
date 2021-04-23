import json

path = '/Users/ericnguyen/Desktop/claire_data/webtool_data/allChestCompression.json'

out_path = '/Users/ericnguyen/Desktop/claire_data/webtool_data/allChestCompression_videoID_to_class.json'

activity = 'chest compression'

with open(path) as f:
    video_map = json.load(f)

    new_map = {}

    links = video_map[activity]

    for link in links:

        new_map[link] = activity

    with open(out_path, 'w') as out_f:
        json.dump(new_map, out_f, indent=4)
