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
    contours=True,
    color="blue",
    redshift=None,
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

    contours : bool, optional
        use fgivenx.plot_contours, else fgivenx.plot_lines

    color : str, optional
        Color of lines.

    redshift : None | float
        Plot against redshift, and if so, what redshift to go back to.

    **kwargs : passed to fgivenx.plot_contours or fgivenx.plot_lines

    Returns
    -------
    ax : matplotlib.Axes

    """
    if ax is None:
        _, ax = plt.subplots()

    pattern = re.compile(r"^[wa]\d+$|^wn$|^Nw$")
    keys = [key for key in list(samples.columns.get_level_values(0))
            if pattern.match(key)]
    n = max(int(key[1:]) for key in keys if key != "wn" and key != "Nw") + 2
    # regex matching may pick up the wrong order of keys, so get the correct
    # order from the relevant theory
    if "Nw" in keys:
        theory = darkknot.Adaptive({"n": n})
        keys = theory.params.keys()
        keys = list(filter(lambda k: k in samples, keys))
    else:
        theory = darkknot.Vanilla({"n": n})
        keys = theory.params.keys()

    if redshift:
        def f(z, theta):
            return theory.flexknot(1/z+1, theta)
        # really z
        a = np.linspace(1/theory.atoday-1, redshift)
        # change xlabel if it is still the default
        if xlabel == r"$a$":
            xlabel = r"$z$"
    else:
        f = theory.flexknot
        a = np.linspace(theory.amin, theory.atoday, resolution),

    if contours:
        plot_contours(
            f,
            a,
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            colors=colors,
            **kwargs,
        )
    else:
        plot_lines(
            f,
            a,
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            color=color,
            **kwargs,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax
