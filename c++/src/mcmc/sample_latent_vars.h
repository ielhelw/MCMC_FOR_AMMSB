#ifndef MCMC_SAMPLE_LATENT_VARS_H__
#define MCMC_SAMPLE_LATENT_VARS_H__

#include <vector>
#include <cstring>  // for size_t

int sample_z_ab_from_edge(bool y, const std::vector<double> &pi_a,
                          const std::vector<double> &pi_b,
                          const std::vector<double> &beta, double epsilon,
                          ::size_t K);

#endif  // MCMC_SAMPLE_LATENT_VARS_H__
