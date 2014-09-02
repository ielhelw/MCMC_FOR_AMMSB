#ifndef MCMC_SAMPLE_LATENT_VARS_H__
#define MCMC_SAMPLE_LATENT_VARS_H__

#include <cmath>

#include "mcmc/exception.h"
#include "mcmc/random.h"

namespace mcmc {

/**
 * we need to calculate z_ab. We can use deterministic way to calculate this
 * for each k,  p[k] = p(z_ab=k|*) = \sum_{i}^{} p(z_ab=k, z_ba=i|*)
 * then we simply sample z_ab based on the distribution p.
 * this runs in O(K)
 */
int sample_z_ab_from_edge(int y, const std::vector<double> &pi_a,
						  const std::vector<double> &pi_b,
						  const std::vector<double> &beta,
						  double epsilon, ::size_t K) {
	std::vector<double> p(K);
    std::vector<double> bounds(K);

    for (::size_t i = 0; i < K; i++) {
        double tmp = std::pow(beta[i], y) * std::pow(1.0 - beta[i], 1.0 - y) * pi_a[i] * pi_b[i];
        tmp += std::pow(epsilon, y) * std::pow(1.0 - epsilon, 1.0 - y) * pi_a[i] * (1.0 - pi_b[i]);
        p[i] = tmp;
	}

    bounds[0] = p[0];
    for (::size_t k = 1; k < K; k++) {
        bounds[k] = bounds[k-1] + p[k];
	}

    double location = Random::random->random() * bounds[K-1];
    // get the index of bounds that containing location.
    for (::size_t i = 0; i < K; i++) {
		if (location <= bounds[i]) {
			return (int)i;
		}
	}

    // failed, should not happen!
    return -1;
}

}	// namespace mcmc

#endif	// ndef MCMC_SAMPLE_LATENT_VARS_H__
