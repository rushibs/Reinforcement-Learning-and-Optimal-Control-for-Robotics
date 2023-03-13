#Find the optimal cost-to-go given optimal sequence of actions

import math
import numpy as np


def next_state(x, u):
    f = x + 0.1*u
    if f <= 5 and f >= 0:
        next_state = f
    elif f > 5:
        next_state = 5
    else:
        next_state = 0

    return next_state
        