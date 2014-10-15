#ifndef MCMC_LEARNING_MCMC_HOST_SAMPLER_H__
#define MCMC_LEARNING_MCMC_HOST_SAMPLER_H__

#include "mcmc/learning/learner.h"
#include "mcmc/learning/mcmc_sampler.h"


namespace mcmc {
namespace learning {

/**
 * MCMC Sampler, common base class for the Host implementations
 */
class MCMCHostSampler : public MCMCSampler {

public:
    MCMCHostSampler(const Options &args, const Network &network,
					::size_t num_node_sample = 0,
					double eta0 = 0.0, double eta1 = 0.0)
			: MCMCSampler(args, network, num_node_sample, eta0, eta1) {
	}


    virtual ~MCMCHostSampler() {
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

};

} // namespace learning
} // namespace mcmc


#endif	// ndef MCMC_LEARNING_MCMC_HOST_SAMPLER_H__
