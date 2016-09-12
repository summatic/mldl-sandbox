# __Author__: Hanseok Jo
import math


def sigm(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def diff_sigm(x):
    return sigm(x) * (1 - sigm(x))


def diff_relu(x):
    if x > 0:
        return 1
    else:
        return 0
