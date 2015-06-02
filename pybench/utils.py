from collections import OrderedDict
from itertools import product


def value_combinations(dct):
    """Return a list of dictionaries with all combinations of values from
    dictionary `dct`, where each value is a list. The input ::

        {'a': [1, 2], 'b': [3, 4, 5]}

    yields ::

        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5},
         {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]
    """
    dct = dict(dct)
    keys = sorted(dct)
    return [OrderedDict(zip(keys, p)) for p in product(*(dct[k] for k in keys))]
