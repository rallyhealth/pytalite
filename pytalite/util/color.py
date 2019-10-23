from random import seed, shuffle
import numpy as np
from matplotlib.colors import ListedColormap

__all__ = ['main', 'auxiliary', 'get_colors', 'gen_cmap']

seed(1024)

c1 = (146, 190, 193)
c2 = (111, 171, 205)
c3 = (186, 133, 141)
c4 = (252, 150, 127)
c5 = (255, 173, 126)
c6 = (255, 193, 94)
c7 = (254, 110, 0)

main = list(map(lambda x: tuple([x[i] / 255 for i in range(3)]), [c1, c2, c3, c4, c5, c6, c7]))


def _interpolate(ca, cb, k):
    return tuple([(1 - k) * ca[i] + k * cb[i] for i in range(3)])


def _compute_auxiliary_colors():
    ks = [0.2, 0.4, 0.6, 0.8]
    new_colors = set()

    for ca, cb in zip(main[:-1], main[1:]):
        for k in ks:
            new_colors.add(_interpolate(ca, cb, k))

    result = sorted(new_colors)
    shuffle(result)
    return result


auxiliary = _compute_auxiliary_colors()


def get_colors(n):
    if n <= len(main):
        return main[:n]
    elif n <= len(main) + len(auxiliary):
        return main + auxiliary[:n - len(main)]
    else:
        return main + auxiliary + get_colors(n - len(main) - len(auxiliary))


def gen_cmap(colors, n_entries):
    if len(n_entries) != len(colors) - 1:
        raise ValueError("n_entries must have a length of (number of colors - 1)")

    result = []

    for (ca, cb), n_entry in zip(zip(colors[:-1], colors[1:]), n_entries):
        ks = np.linspace(0, 1, n_entry)
        result += [_interpolate(ca, cb, k) for k in ks]

    return ListedColormap(np.array(result))
