from random import Random as SystemRandom
from com.uva.custom_random.custom_random import CustomRandom

class WrapperRandom:

    def __init__(self):
        self.random_sources = (
            "graph init",
            "create field",
            "minibatch sampler",
            "neighbor sampler",
            "update phi",
            "update beta",
        )
        self.seed(0, False)

    def seed(self, seed = 0, use_mcmc_random = False):
        self.USE_MCMC_RANDOM = use_mcmc_random

        if self.USE_MCMC_RANDOM:
            i = 0
            self.custom_rng = { }
            for s in self.random_sources:
                self.custom_rng[s] = CustomRandom(seed + i)
                i += 1
        else:
            self.rng = SystemRandom(seed)

    def get(self, source):
        if self.USE_MCMC_RANDOM:
            return self.custom_rng[source]
        else:
            return self.rng

WrapperRandom = WrapperRandom()
# _inst.init(42, True)
