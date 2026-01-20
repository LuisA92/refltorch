import numpy as np
import seaborn as sns


def get_cube_helix_map(n):
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    cmap_list = cmap(np.linspace(0.0, 1, n, retstep=True)[0])
    return cmap_list
