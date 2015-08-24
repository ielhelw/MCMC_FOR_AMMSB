#ifndef MCMC_ESTIMATE_PHI_H__
#define MCMC_ESTIMATE_PHI_H__

#include <vector>

#include "mcmc/config.h"
#include "mcmc/data.h"

namespace mcmc {

/**
 * @result resize and fill [ phi_ab, phi_ba ]
 */
void sample_latent_vars_for_each_pair(
    int a, int b, const std::vector<double> &gamma_a,
    const std::vector<double> &gamma_b,
    const std::vector<std::vector<double> > &lamda, ::size_t K,
    double update_threshold, double epsilon, ::size_t online_iterations,
    const NetworkGraph &linked_edges, std::vector<double> *phi_ab,
    std::vector<double> *phi_ba);
}  // namespace mcmc

#endif  // ndef MCMC_ESTIMATE_PHY_H__
