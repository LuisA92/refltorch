import numpy as np
import seaborn as sns


def get_cube_helix_map(
    n,
    start: float = 0.5,
    rot=-0.55,
    dark=0,
    light: float = 0.8,
    as_cmap: bool = True,
    reverse: bool = False,
):
    cmap = sns.cubehelix_palette(
        start=start,
        rot=rot,
        dark=dark,
        light=light,
        as_cmap=as_cmap,
        reverse=reverse,
    )
    cmap_list = cmap(np.linspace(0.0, 1, n, retstep=True)[0])
    return cmap_list
