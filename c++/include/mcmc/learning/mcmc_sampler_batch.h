#ifndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
#define MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__

#include <cmath>

#include <utility>

#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

// typedef std::set<int>	NodeSet;
typedef std::unordered_set<int>	NodeSet;

/**
 * MCMC Sampler for batch learning. Every update go through the whole data sets.
 */
class MCMCSamplerBatch : public Learner {

public:
    MCMCSamplerBatch(const Options &args, Const Network &graph)
			: Learner(args, graph) {

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning
        num_node_sample = static_cast<::size_t>(std::sqrt(network.get_num_nodes()));

        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
        random.gamma(&theta, eta[0], eta[1], K, 2);		// parameterization for \beta
        random.gamma(&phi, 1, 1, N, K);					// parameterization for \pi

		// FIXME RFHH -- code sharing with variational_inf*::update_pi_beta()
        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0],size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		np::row_normalize(&pi, phi);
	}


	// FIXME make NodeSet a result parameter
    NodeSet sample_neighbor_nodes_batch(int node) {
        NodeSet neighbor_nodes;
		for (int i = 0; i < (int)N; i++) {
			Edge edge(std::min(node, i), std::max(node, i));
			if (network.get_held_out_set().find(edge) == network.get_held_out_set().end() &&
					network.get_test_set().find(edge) == network.get_test_set().end()) {
                neighbor_nodes.insert(i);
			}
		}

        return neighbor_nodes;
	}


	/**
	 * update pi for current node i.
	 */
    void update_pi_for_node(int i, const std::vector<double> &z, std::vector<std::vector<double> > *phi_star, <TYPE3> n) {
        // update gamma, only update node in the grad
		double eps_t;

        if ! stepsize_switch {
            eps_t = std::pow(1024 + step_count, -0.5);
		} else {
            eps_t  = a * std::pow(1 + step_count / b, -c);
		}

        double phi_i_sum = np::sum(phi[i]);
        double noise = random.randn(K);                                 // random noise.

        // get the gradients
		// grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
		std::vector<double> grads(K, -n / phi_i_sum);		// Hard to grasp... RFHH
        for (::size_t k = 0; k < K; k++) {
            grads[k] += 1 / phi[i,k] * z[k];
		}

        // update the phi
        for (::size_t k = 0; k < K; k++) {
			// FIXME RFHH a**0.5 * b**0.5 better written as sqrt(a*b) ?
            phi_star[i][k] = std::abs(phi[i,k] + eps_t/2 * (alpha - phi[i,k] +
														   	grads[k]) +
									  std::pow(eps_t, .5) * std::pow(phi[i,k], .5) * noise[k]);
		}
	}


    void update_beta() {
		std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));
        // sums = np.sum(self.__theta,1)
		std::vector<double> sums(theta.size());
		std::transform(theta.begin(), theta.end(), sums.begin(), np::AddColumn<double>(1));

        // update gamma, only update node in the grad
		double eps_t;
        if (! self.stepsize_switch) {
            eps_t = std::pow(1024+self._step_count, -0.5);
		} else {
            eps_t = a * std::pow(1 + self._step_count / b, -c);
		}

        for (::size_t i = 0; i < N; i++) {
            for (::size_t j = i + 1; j < N; j++) {
				Edge edge(i, j);

				if (in(network.get_held_out_set(), edge) ||
						in(network.get_test_test_set(), edge)) {
                    continue;
				}

                int y_ab = 0;
                if (i,j) in self._network.get_linked_edges():
                    y_ab = 1

                z = self.__sample_z_for_each_edge(y_ab, self._pi[i], self._pi[j], \
                                          self._beta, self._K)
                if z == -1:
                    continue

                grads[z,0] += abs(1-y_ab)/self.__theta[z,0] - 1/ sums[z]
                grads[z,1] += abs(-y_ab)/self.__theta[z,1] - 1/sums[z]


        // update theta
        noise = random.randn(self._K, 2)
        theta_star = copy.copy(self.__theta)
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self.__theta[k,i] + eps_t/2 * (self._eta[i] - self.__theta[k,i] + \
                                      grads[k,i]) + eps_t**.5*self.__theta[k,i] ** .5 * noise[k,i])

        if  self._step_count < 50000:
            self.__theta = theta_star
        else:
            self.__theta = theta_star * 1.0/(self._step_count) + (1-1.0/(self._step_count))*self.__theta
        //self.__theta = theta_star
        // update beta from theta
        temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        self._beta = temp[:,1]

    def __sample_z_for_each_edge(self, y, pi_a, pi_b, beta, K):
        /**
		 * sample latent variables z_ab and z_ba
         * but we don't need to consider all of the cases. i.e  (z_ab = j, z_ba = q) for all j and p.
         * because of the gradient depends on indicator function  I(z_ab=z_ba=k), we only need to consider
         * K+1 different cases:  p(z_ab=0, z_ba=0|*), p(z_ab=1,z_ba=1|*),...p(z_ab=K, z_ba=K|*),.. p(z_ab!=z_ba|*)
		 *
		 * Arguments:
         *   y:        observation [0,1]
         *   pi_a:     community membership for node a
         *   pi_b:     community membership for node b
         *   beta:     community strengh.
         *   epsilon:  model parameter.
         *   K:        number of communities.
		 *
		 * Returns the community index. If it falls into the case that z_ab!=z_ba, then return -1
         */
        p = np.zeros(K+1)
        for k in range(0,K):
            p[k] = beta[k]**y*(1-beta[k])**(1-y)*pi_a[k]*pi_b[k]
        p[K] = 1 - np.sum(p[0:K])

        // sample community based on probability distribution p.
        bounds = np.cumsum(p)
        location = random.random() * bounds[K]

        // get the index of bounds that containing location.
        for i in range(0, K):
                if location <= bounds[i]:
                    return i
        return -1


    def __sample_latent_vars(self, node, neighbor_nodes):
        /**
		 * given a node and its neighbors (either linked or non-linked), return the latent value
		 * z_ab for each pair (node, neighbor_nodes[i].
         */
        z = np.zeros(self._K)
        for neighbor in neighbor_nodes:
            y_ab = 0      // observation
            if (min(node, neighbor), max(node, neighbor)) in self._network.get_linked_edges():
                y_ab = 1

            z_ab = sample_z_ab_from_edge(y_ab, self._pi[node], self._pi[neighbor], self._beta, self._epsilon, self._K)
            z[z_ab] += 1

        return z


    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        """ run mini-batch based MCMC sampler """
        while self._step_count < self._max_iteration and not self._is_converged():
            //print "step: " + str(self._step_count)
            ppx_score = self._cal_perplexity_held_out()
            print str(ppx_score)
            self._ppxs_held_out.append(ppx_score)

            phi_star = copy.copy(self._pi)
            // iterate through each node, and update parameters pi_a
            for i in range(0, self._N):
                // update parameter for pi_i
                //print "updating: " + str(i)
                neighbor_nodes = self.__sample_neighbor_nodes_batch(i)
                z = self.__sample_latent_vars(i, neighbor_nodes)
                self.__update_pi_for_node(i, z, phi_star, len(neighbor_nodes))

            self.__phi = phi_star
            self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]

            // update beta
            z = self.__update_beta()

            self._step_count += 1

        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

protected:
	double	a;
	double	b;
	double	c;

	::size_t num_node_sample;

	std::vector<std::vector<double> > theta;		// parameterization for \beta
	std::vector<std::vector<double> > phi;			// parameterization for \pi
};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
