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
	// UNUSED: std::vector<double> p(K, 0.0);

#ifdef EFFICIENCY_FOLLOWS_PYTHON

	// FIXME: selection based on y, done with pow()
	// FIXME: lift common subexpr in second statement
    for (::size_t i = 0; i < K; i++) {
        double tmp = std::pow(beta[i], y) * std::pow(1.0 - beta[i], 1 - y) * pi_a[i] * pi_b[i];
        tmp += std::pow(epsilon, y) * std::pow(1.0 - epsilon, 1 - y) * pi_a[i] * (1.0 - pi_b[i]);
        p[i] = tmp;
	}
#else
	if (y == 1) {
		for (::size_t i = 0; i < K; i++) {
			double tmp = beta[i] * pi_a[i] * pi_b[i];
			tmp += epsilon * pi_a[i] * (1.0 - pi_b[i]);
			p[i] = tmp;
		}
	} else {
		for (::size_t i = 0; i < K; i++) {
			double tmp = (1.0 - beta[i]) * pi_a[i] * pi_b[i];
			tmp += (1.0 - epsilon) * pi_a[i] * (1.0 - pi_b[i]);
			p[i] = tmp;
		}
	}
#endif

    for (::size_t k = 1; k < K; k++) {
        p[k] += p[k - 1];
	}

    double r = Random::random->random();
    double location = r * p[K-1];
    // get the index of bounds that containing location.
    for (::size_t i = 0; i < K; i++) {
		if (location <= p[i]) {
			return (int)i;
		}
	}

    // failed, should not happen!
    return -1;
}

}	// namespace mcmc

#endif	// ndef MCMC_SAMPLE_LATENT_VARS_H__
