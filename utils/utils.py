import json


def read_json(path):

    with open(path) as f:
        data_dict = json.load(f)

    return data_dict


def write_json(data_dict, path):

    '''

    :param data_dict: dict to write to json
    :param path: path to json
    :param indent: for easy viewing.  Use None if you want to save a lot of space
    :return:
    '''

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def write_list(data_list, path):

    with open(path, "w") as f:
        for item in data_list:
            f.write("%s\n" % item)
