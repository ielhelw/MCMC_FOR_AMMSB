#ifndef MCMC_WRAPPER_RANDOM_H__
#define MCMC_WRAPPER_RANDOM_H__

#include "mcmc/exception.h"
#include "mcmc/random.h"

namespace mcmc {

class SourceAwareRandom {

public:
	enum RANDOM_SOURCE {
		GRAPH_INIT,
		THETA_INIT,
		PHI_INIT,
		MINIBATCH_SAMPLER,
		NEIGHBOR_SAMPLER,
		PHI_UPDATE,
		BETA_UPDATE,
		NUM_RANDOM_SOURCE,
	};

	void Init(int seed) {
#ifdef SOURCE_AWARE_RANDOM
		for (int i = 0; i < NUM_RANDOM_SOURCE; ++i) {
			rng_.push_back(new Random::Random(seed + i + 1, seed + i, true));
		}
#else
		rng_.push_back(new Random::Random(seed + 1, seed, false));
#endif
	}

	~SourceAwareRandom() {
		for (auto & r : rng_) {
			delete r;
		}
	}

	Random::Random *random(RANDOM_SOURCE source) {
#ifdef SOURCE_AWARE_RANDOM
		return rng_[source];
#else
		return rng_[0];
#endif
	}

	int seed(int i) const {
		return rng_[0]->seed(i);
	}

protected:
	std::vector<Random::Random *> rng_;
};

}	// namespace mcmc

#endif	// ndef MCMC_WRAPPER_RANDOM_H__
