#
# utils.py
#

import numpy as np


def get_prec(val):
    """
    Gets the precision of the val passed in.

    :param: val The number to get the precision of
    """
    if np.isnan(val) or val == float('inf') or int(val) == val:
        return 9999999

    split_arr_dot = str(val).split('.')
    if len(split_arr_dot) == 2:
        return len(split_arr_dot[1])
    else:
        split_arr_e = str(val).split('e')
        exp = int(split_arr_e[1])
        return -exp if exp < 0 else 0


def is_int(val):
    return isinstance(val, (int, long))