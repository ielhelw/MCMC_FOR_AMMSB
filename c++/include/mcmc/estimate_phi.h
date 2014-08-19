#ifndef MCMC_ESTIMATE_PHI_H__
#define MCMC_ESTIMATE_PHI_H__

namespace mcmc {

typedef std::pair<std::vector<double>, std::vector<double> >	DoubleVectorPair;

static DoubleVectorPair sample_latent_vars_for_each_pair(int a, int b,
														 const std::vector<double> &gamma_a,
														 const std::vector<double> &gamma_b,
														 const std::vector<std::vector<double> > &lamda,
														 ::size_t K, double update_threshold, double epsilon,
														 ::size_t online_iterations, const EdgeSet *linked_edges) {
	throw UnimplementedException("sample_latent_vars_for_each_pair");
	return DoubleVectorPair(std::vector<double>(0), std::vector<double>(0));
}

}	// namespace mcmc

#endif	// ndef MCMC_ESTIMATE_PHY_H__
