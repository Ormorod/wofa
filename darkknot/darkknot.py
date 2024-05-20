import numpy as np
from cobaya import Theory
from flexknot import AdaptiveKnot, FlexKnot


class DarkKnot(Theory):
    """
    Abstract base class for flexknot w(a).

    Requires additional class attribute params, which needs to be
    ordered in the correct structure for a flexknot, and definition of
    self.flexknot needs to be added to
    """

    num_as = 10000
    amin = 1e-10
    atoday = 1

    def __init__(self, n, *args, **kwargs):
        if n >= 2:
            self.params["w0"] = None
        for i in range(1, n-1):
            self.params[f"a{i}"] = None
            self.params[f"w{i}"] = None
        if n >= 1:
            self.params["wn"] = None

        return super().__init__(*args, **kwargs)

    def wofa(self, theta):
        a = np.logspace(np.log10(self.amin), np.log10(self.atoday), self.num_as)
        w = self.flexknot(a, theta)

        return a, w

    def calculate(self, state, want_derived=True, **params_values_dict):
        theta = np.array([params_values_dict[p] for p in self.params.keys()])
        a, w = self.wofa(theta)
        state["dark_energy"] = {
            "a": a,
            "w": w,
        }

    def get_dark_energy(self):
        return self.current_state["dark_energy"]


class Adaptive(DarkKnot):

    def __init__(self, n, *args, **kwargs):
        self.params = {"Nw": None}
        self.flexknot = AdaptiveKnot(self.amin, self.atoday)
        super().__init__(n, *args, **kwargs)


class Vanilla(DarkKnot):
    def __init__(self, n, *args, **kwargs):
        self.params = dict(
            [(f"w{i}", None) for i in range(n)]
        )
        self.flexknot = FlexKnot(self.amin, self.atoday)
        super().__init__(n, *args, **kwargs)
