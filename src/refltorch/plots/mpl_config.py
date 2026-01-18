import matplotlib as mpl

CUSTOM_CMAP = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky
]


def setup_mpl_config():
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler("color", CUSTOM_CMAP)
