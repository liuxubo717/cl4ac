import collections
from collections import OrderedDict
from typing import Dict

import torch


class Sample(OrderedDict):
    """Sample represent some arbitrary data. All datasets in MMF must
    return an object of type ``Sample``.

    Args:
        init_dict (Dict): Dictionary to init ``Sample`` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())
