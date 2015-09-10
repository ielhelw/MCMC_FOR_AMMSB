from random import Random as SystemRandom
from com.uva.custom_random.custom_random import CustomRandom
import numpy as np

class NumPyEnabledRandom(SystemRandom):
    def __init__(self, seed):
        SystemRandom.__init__(self, seed)

    def gamma(self, a, b, dims):
        return np.random.gamma(a, b, dims)

    def randn(self, k, k2 = 1):
        return np.random.randn(k, k2)

class SourceAwareRandom:

    def __init__(self):
        self.random_sources = (
            "graph init",
            "theta init",
            "phi init",
            "minibatch sampler",
            "neighbor sampler",
            "phi update",
            "beta update",
        )
        self.seed(0, False)

    def seed(self, seed = 0, use_mcmc_random = False):
        self.USE_MCMC_RANDOM = use_mcmc_random

        if self.USE_MCMC_RANDOM:
            i = 0
            self.custom_rng = { }
            for s in self.random_sources:
                print "random source[" + str(i) + "] for \"" + s + "\""
                self.custom_rng[s] = CustomRandom(seed + i)
                i += 1
        else:
            self.rng = NumPyEnabledRandom(seed)

    def get(self, source):
        if self.USE_MCMC_RANDOM:
            return self.custom_rng[source]
        else:
            return self.rng

    def sample_range(self, source, N, count):
        if self.USE_MCMC_RANDOM:
            return self.custom_rng[source].sample_range(N, count)
        else:
           return self.rng.sample(list(xrange(N)), count)

SourceAwareRandom = SourceAwareRandom()
# _inst.init(42, True)
