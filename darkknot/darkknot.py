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

    params = {
        "Nw": None,
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "a4": None,
        "w4": None,
        "a5": None,
        "w5": None,
        "a6": None,
        "w6": None,
        "a7": None,
        "w7": None,
        "w8": None,
    }

    def __init__(self, *args, **kwargs):

        self.flexknot = AdaptiveKnot(self.amin, self.atoday)
        super().__init__(*args, **kwargs)


class VanillaDarkKnot(DarkKnot):
    def __init__(self, *args, **kwargs):

        self.flexknot = FlexKnot(self.amin, self.atoday)
        super().__init__(*args, **kwargs)


# TODO: see if I can just put the params in the constructor to be
# defined with a for loop


class Vanilla0(VanillaDarkKnot):

    params = {}


class Vanilla1(VanillaDarkKnot):

    params = {
        "w8": None,
    }


class Vanilla2(VanillaDarkKnot):

    params = {
        "w0": None,
        "w8": None,
    }


class Vanilla3(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "w8": None,
    }


class Vanilla4(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "w8": None,
    }


class Vanilla5(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "w8": None,
    }


class Vanilla6(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "a4": None,
        "w4": None,
        "w8": None,
    }


class Vanilla7(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "a4": None,
        "w4": None,
        "a5": None,
        "w5": None,
        "w8": None,
    }


class Vanilla8(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "a4": None,
        "w4": None,
        "a5": None,
        "w5": None,
        "a6": None,
        "w6": None,
        "w8": None,
    }


class Vanilla9(VanillaDarkKnot):

    params = {
        "w0": None,
        "a1": None,
        "w1": None,
        "a2": None,
        "w2": None,
        "a3": None,
        "w3": None,
        "a4": None,
        "w4": None,
        "a5": None,
        "w5": None,
        "a6": None,
        "w6": None,
        "a7": None,
        "w7": None,
        "w8": None,
    }
