import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples
from fgivenx import plot_contours
from darkknot import darkknot

theory_list = [
    darkknot.Vanilla0,
    darkknot.Vanilla1,
    darkknot.Vanilla2,
    darkknot.Vanilla3,
    darkknot.Vanilla4,
    darkknot.Vanilla5,
    darkknot.Vanilla6,
    darkknot.Vanilla7,
    darkknot.Vanilla8,
    darkknot.Vanilla9,
    darkknot.Adaptive,
]


def plot(
    samples: NestedSamples,
    ax=None,
    resolution=100,
    colors="Blues_r",
    fig=None,
    xlabel=r"$a$",
    ylabel=r"$w(a)$",
    **kwargs,
):
    """
    Plot functional posterior of w(a) samples.

    **kwargs passed on to fgivenx.plot_contours (or plot_lines)
    """
    if ax is None:
        _, _ax = plt.subplots()
    else:
        _ax = ax

    # special case to allow Nw column to be added to samples, to treat
    # concatenated Vanilla samples to be treated as Adaptive, even if
    # they don't go up to 9 nodes
    if "Nw" in samples:
        theory = darkknot.Adaptive()
        keys = theory.params.keys()
        keys = list(filter(lambda k: k in samples, keys))
    else:
        for Theory in theory_list[::-1]:
            if all([key in samples for key in Theory.params.keys()]):
                theory = Theory()
                break
        keys = theory.params.keys()

    cbar = plot_contours(
        lambda a, theta: theory.flexknot(a, theta),
        np.linspace(theory.amin, theory.atoday, resolution),
        samples[keys],
        weights=samples.get_weights(),
        ax=_ax,
        colors=colors,
    )
    _ax.set(xlabel=xlabel, ylabel=ylabel)

    return _ax
