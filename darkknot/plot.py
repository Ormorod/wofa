import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples
from fgivenx import plot_contours
from darkknot import darkknot

theory_list = [
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

xlabel = r"$a$"
ylabel = r"$w(a)$"


def plot(samples: NestedSamples, ax=None, resolution=100, colors="Blues_r", title=None, fig=None):
    if ax is None:
        _, _ax = plt.subplots()
    else:
        _ax = ax

    for Theory in theory_list[::-1]:
        if all([key in samples for key in Theory.params.keys()]):
            break
    print(Theory)
    theory = Theory()

    weights = np.array([idx[1] for idx in samples.index])

    a = np.linspace(theory.amin, theory.amax, resolution)

    cbar = plot_contours(
        lambda a, theta: theory.flexknot(a, theta),
        ks,
        samples[theory.params.keys()],
        weights=weights,
        ax=_ax,
        colors=colors,
    ) 

    return _ax


