import json



def read_json(path):

    try:
        with open(path) as f:
            data_dict = json.load(f)

    except Exception as e:
        print(e)
        return False

    return data_dict


def write_json(data_dict, path, indent=4):

    '''

    :param data_dict: dict to write to json
    :param path: path to json
    :param indent: for easy viewing.  Use None if you want to save a lot of space
    :return:
    '''

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=indent)

    except Exception as e:
        print(e)
        return False

    return True


def write_list(data_list, path):

    with open(missing_path, "w") as f:
        try:
            for item in missing_list:
                f.write("%s\n" % item)

        except Exception as e:
            print(e)
            return False

    return True