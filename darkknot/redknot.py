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
    n = None
    params = {}

    def initialize(self):
        if self.n >= 1:
            self.params["wn"] = None
        for i in range(self.n - 2, 0, -1):
            self.params[f"a{i}"] = None
            self.params[f"w{i}"] = None
        if self.n >= 2:
            self.params["w0"] = None

        return super().initialize()

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

    n = 9

    def __init__(self, *args, **kwargs):
        self.flexknot = AdaptiveKnot(self.amin, self.atoday)
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.params = {"Nw": None}
        super().initialize()


class Vanilla(DarkKnot):
    def __init__(self, *args, **kwargs):
        self.flexknot = FlexKnot(self.amin, self.atoday)
        super().__init__(*args, **kwargs)
