import yaml
from dotmap import DotMap


def extend_dict(extend_me, extend_by):
    if isinstance(extend_me, dict):
        for k, v in extend_by.iteritems():
            if k in extend_me:
                extend_dict(extend_me[k], v)
            else:
                extend_me[k] = v
    else:
        if isinstance(extend_me, list):
            extend_list(extend_me, extend_by)
        else:
            extend_me += extend_by


def extend_list(extend_me, extend_by):
    missing = []
    for item1 in extend_me:
        if not isinstance(item1, dict):
            continue

        for item2 in extend_by:
            if not isinstance(item2, dict) or item2 in missing:
                continue
            extend_dict(item1, item2)


def get_config(path):
    with open(path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    with open('config/base.yml', 'r') as file:
        base_configuration = yaml.load(file, Loader=yaml.FullLoader)
    configuration = DotMap(configuration)
    base_configuration = DotMap(base_configuration)
    extend_dict(configuration, base_configuration)
    return configuration


if __name__ == '__main__':
    config = get_config('config/bert-base.yml')
