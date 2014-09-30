#ifndef MCMC_LEARNING_MCMC_SAMPLER_H__
#define MCMC_LEARNING_MCMC_SAMPLER_H__


#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

class MCMCSampler : virtual public Learner {
public:
    MCMCSampler(const Options &args, const Network &graph,
			   	::size_t num_node_sample = 0,
			   	double eta0 = 0.0, double eta1 = 0.0)
			: Learner(args, graph) {

        // control parameters for learning

		if (num_node_sample == 0) {
			this->num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
		} else {
			this->num_node_sample = num_node_sample;
		}

        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		if (eta0 != 0.0 || eta1 != 0.0) {
			std::cerr << "Ignore eta[] in random.gamma: use " << eta0 << " and " << eta1 << std::endl;
		} else {
			eta0 = eta[0];
			eta1 = eta[1];
		}
		// theta = Random::random->gamma(eta[0], eta[1], K, 2);
		theta = Random::random->gamma(eta0, eta1, K, 2);
		phi = Random::random->gamma(1, 1, N, K);

        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);

		info(std::cout);

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;
	}

	virtual ~MCMCSampler() {
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}


protected:
	// replicated in both mcmc_sampler_
	double	a;
	double	b;
	double	c;

	::size_t num_node_sample;

	std::vector<std::vector<double> > theta;		// parameterization for \beta
	std::vector<std::vector<double> > phi;			// parameterization for \pi

};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_H__
