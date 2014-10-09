#ifndef MCMC_LEARNING_MCMC_SAMPLER_H__
#define MCMC_LEARNING_MCMC_SAMPLER_H__


#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

class MCMCSampler : public Learner {
public:
    MCMCSampler(const Options &args, const Network &network,
			   	::size_t num_node_sample = 0,
			   	double eta0 = 0.0, double eta1 = 0.0)
			: Learner(args, network) {

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning

		if (num_node_sample == 0) {
			this->num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
		} else {
			this->num_node_sample = num_node_sample;
		}
		std::cerr << "num_node_sample " << num_node_sample << std::endl;

        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		if (eta0 == 0.0 && eta1 == 0.0) {
			eta0 = args.eta0;
			eta1 = args.eta1;
		} else {
			std::cerr << "Override eta: use (" << eta0 << "," << eta1 << ") i.s.o. command line" << std::endl;
		}
		theta = Random::hostRandom->gamma(eta0, eta1, K, 2);
		phi = Random::hostRandom->gamma(1, 1, N, K);

        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);

		info(std::cout);
	}

	virtual ~MCMCSampler() {
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}


	// Data dependencies:
	// 		pi[node][*]
	// 		beta
	// 		network::linked_edges
	// 		inherit from sample_z_ab_from_edge:
	// 			pi[node] pi[neighbor] beta
    std::vector<int> sample_latent_vars(int node, const OrderedVertexSet &neighbor_nodes) const {
        /**
        given a node and its neighbors (either linked or non-linked), return the latent value
        z_ab for each pair (node, neighbor_nodes[i].
         */
		std::vector<int> z(K, 0);
		// std::cerr << "node " << node << " " << neighbor_nodes.size() << std::endl;
        for (auto neighbor = neighbor_nodes.begin();
			 	neighbor != neighbor_nodes.end();
				neighbor++) {
            int y_ab = 0;      // observation
			Edge edge(std::min(node, *neighbor), std::max(node, *neighbor));
            if (edge.in(network.get_linked_edges())) {
                y_ab = 1;
			}

            int z_ab = sample_z_ab_from_edge(y_ab, pi[node], pi[*neighbor], beta, epsilon, K);
            z[z_ab] += 1;
		}

        return z;
	}


    OrderedVertexSet nodes_in_batch(const OrderedEdgeSet &mini_batch) const {
        /**
        Get all the unique nodes in the mini_batch.
         */
        OrderedVertexSet node_set;
        for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
            node_set.insert(edge->first);
            node_set.insert(edge->second);
		}

        return node_set;
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
