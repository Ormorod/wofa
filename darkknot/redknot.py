import numpy as np
from cobaya import Theory
from flexknot import AdaptiveKnot, FlexKnot


class RedKnot(Theory):
    """
    Abstract base class for flexknot w(z).

    Requires additional class attribute params, which needs to be
    ordered in the correct structure for a flexknot, and definition of
    self.flexknot needs to be added to
    """

    num_zs = 10000
    zmax = 1e10
    ztoday = 0
    n = None
    params = {}

    def initialize(self):
        if self.n >= 2:
            self.params["w0"] = None
        for i in range(1, self.n - 1):
            self.params[f"z{i}"] = None
            self.params[f"w{i}"] = None
        if self.n >= 1:
            self.params["wn"] = None

        return super().initialize()

    def wofa(self, theta):
        oneplusz = np.logspace(np.log10(1+self.ztoday),
                               np.log10(1+self.zmax),
                               self.num_zs)[::-1]
        a = 1 / oneplusz
        w = self.flexknot(oneplusz-1, theta)

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


class RedAdaptive(RedKnot):

    n = 9

    def __init__(self, *args, **kwargs):
        self.flexknot = AdaptiveKnot(self.ztoday, self.zmax)
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.params = {"Nw": None}
        super().initialize()


class Strawberry(RedKnot):
    def __init__(self, *args, **kwargs):
        self.flexknot = FlexKnot(self.ztoday, self.zmax)
        super().__init__(*args, **kwargs)
