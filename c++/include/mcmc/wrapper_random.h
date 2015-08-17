#ifndef MCMC_WRAPPER_RANDOM_H__
#define MCMC_WRAPPER_RANDOM_H__

#include "mcmc/random.h"

namespace mcmc {

class WrapperRandom {

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

	WrapperRandom(int seed) {
#ifdef SOURCE_AWARE_RANDOM
		for (int i = 0; i < NUM_RANDOM_SOURCE; ++i) {
			_rng.push_back(new Random::Random(seed + i, true));
		}
#else
		_rng.push_back(new Random::Random(seed, false));
#endif
	}

	~WrapperRandom() {
		for (auto & r : _rng) {
			delete r;
		}
	}

	Random::Random *random(RANDOM_SOURCE source) {
#ifdef SOURCE_AWARE_RANDOM
		return _rng[source];
#else
		return _rng[0];
#endif
	}

protected:
	bool _source_aware;
	std::vector<Random::Random *> _rng;
};

extern WrapperRandom *wrapper_random;

}	// namespace mcmc

#endif	// ndef MCMC_WRAPPER_RANDOM_H__
