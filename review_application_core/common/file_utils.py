import json


def write_json(SAVE_JSON_DIR, args):
    with open(SAVE_JSON_DIR, 'w') as f:
        json.dump(vars(args), f, indent=4)


def read_json(SAVE_JSON_DIR):
    with open(SAVE_JSON_DIR) as json_file:
        return json.load(json_file)
