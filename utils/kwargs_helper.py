def get_kwargs_value(kwargs, key, default=None):
    if key in kwargs.keys():
        return kwargs[key]
    return default
