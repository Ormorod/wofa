import re
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples
from fgivenx import plot_contours, plot_lines
from darkknot import darkknot


def plot(
    samples: NestedSamples,
    ax=None,
    resolution=100,
    colors="Blues_r",
    xlabel=r"$a$",
    ylabel=r"$w(a)$",
    lines=False,
    color="blue",
    **kwargs,
):
    """
    Plot functional posterior of w(a) samples.

    Parameters
    ----------
    samples : NestedSamples
        Samples to plot.

    ax : matplotlib.Axes, optional
        Axes to plot on. If None, a new figure is created.

    resolution : int, optional
        Resolution of the plot.

    colors : str, optional
        Color map to use for contours.

    xlabel : str, optional
        Label for x-axis.

    ylabel : str, optional
        Label for y-axis.

    lines : bool, optional
        Plot lines instead of contours.

    color : str, optional
        Color of lines.

    **kwargs : passed to fgivenx.plot_contours or fgivenx.plot_lines

    Returns
    -------
    ax : matplotlib.Axes

    """
    if ax is None:
        _, ax = plt.subplots()

    # special case to allow Nw column to be added to samples, to treat
    # concatenated Vanilla samples to be treated as Adaptive, even if
    # they don't go up to 9 nodes
    if "Nw" in samples:
        theory = darkknot.Adaptive()
        keys = theory.params.keys()
        keys = list(filter(lambda k: k in samples, keys))
    else:
        pattern = re.compile(r"^[wa]\d+$|^wn$")
        keys = (key for key in list(samples.columns.get_level_values(0))
                if pattern.match(key))
        print(f"regexed {keys}")
        n = max(int(key[1:]) for key in keys if key != "wn") + 2
        theory = darkknot.Vanilla(n)
        keys = theory.params.keys()
        print(f"theory keys: {keys}")

    if lines:
        plot_lines(
            lambda a, theta: theory.flexknot(a, theta),
            np.linspace(theory.amin, theory.atoday, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            color=color,
            **kwargs,
        )
    else:
        plot_contours(
            lambda a, theta: theory.flexknot(a, theta),
            np.linspace(theory.amin, theory.atoday, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            colors=colors,
            **kwargs,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax
