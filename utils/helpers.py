import os
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

MISSING_VAL_FILLER = -1000


def convert_str_to_list(item):
    """
    Converts a string to a list if possible. Useful when reading a DataFrame from a CSV file,
    where a column should contain lists but these are encoded as strings

    """
    if type(item) == str and '[' in item:
        item_as_list = literal_eval(item)
        return item_as_list
    else:
        return item


def convert_to_set(x):
    if type(x) == list:
        res = set([i for i in x if not is_missing(i)])
        return res
    elif is_missing(x):
        return set()
    else:
        return set(x)


def is_missing(s):
    return s is None or s == '' or pd.isnull(s) or s == set([])


def is_equal(s1, s2):
    """Compare two strings. If both 'exist' then return 1 if they are equal and -1 otherwise. Otherwise return 0."""
    if is_missing(s1) or is_missing(s2):
        return 0
    elif s1 == s2:
        return 1
    else:
        return -1


def compute_diff(num_1, num_2):
    try:
        return np.abs(num_1 - num_2)
    except TypeError:
        return None


def load_pickle_file(path):
    with open(path, 'rb') as filehandle:
        # read the data as binary data stream
        o = pickle.load(filehandle)
    return o


def dump_pickle_file(path, object):
    with open(path, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(object, filehandle)



def extract_file_name_from_path(filepath):
    """https://shakyaabiral.wordpress.com/2016/07/15/python-extract-filename-and-extension-from-filepath/"""
    filename_w_ext = os.path.basename(filepath)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return filename, file_extension


def cosine_sim(v1, v2):
    return 1 - np.round(cosine(v1, v2), 3)

