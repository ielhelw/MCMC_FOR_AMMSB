#ifndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
#define MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/sample_latent_vars.h"

#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {


template <typename T>
class MCMCMyOp {
public:
	MCMCMyOp(T a, T b) : a(a), b(b) {
	}

	T operator() (const T &x, const T &y) {
		return x * a + y * b;
	}

protected:
	T a;
	T b;
};


/**
 * MCMC Sampler for batch learning. Every update go through the whole data sets.
 */
class MCMCSamplerBatch : public Learner {

public:
    MCMCSamplerBatch(const Options &args, const Network &graph)
			: Learner(args, graph) {

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning
        num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));

        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		theta = Random::random->gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
		phi = Random::random->gamma(1, 1, N, K);					// parameterization for \pi

		// FIXME RFHH -- code sharing with variational_inf*::update_pi_beta()
        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);
	}


    virtual ~MCMCSamplerBatch() {
	}


	// FIXME make VertexSet a result parameter
    OrderedVertexSet sample_neighbor_nodes_batch(int node) const {
        OrderedVertexSet neighbor_nodes;
		for (int i = 0; i < (int)N; i++) {
			Edge edge(std::min(node, i), std::max(node, i));
			if (! edge.in(network.get_held_out_set()) && ! edge.in(network.get_test_set())) {
				if (false && i == node) {
					std::cerr << "Ooppssss.... is a self-cycle OK? " << i << std::endl;
				}
                neighbor_nodes.insert(i);
			}
		}

        return neighbor_nodes;
	}


	/**
	 * update pi for current node i.
	 */
    void update_pi_for_node(::size_t i, const std::vector<int> &z, std::vector<std::vector<double> > *phi_star, ::size_t n) const {
        // update gamma, only update node in the grad
		double eps_t;

        if (! stepsize_switch) {
            eps_t = std::pow(1024 + step_count, -0.5);
		} else {
            eps_t  = a * std::pow(1 + step_count / b, -c);
		}

        double phi_i_sum = np::sum(phi[i]);
		std::vector<double> noise = Random::random->randn(K);		// random noise.

        // get the gradients
		// grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
		std::vector<double> grads(K, -(int)n / phi_i_sum);		// Hard to grasp... RFHH
        for (::size_t k = 0; k < K; k++) {
            grads[k] += 1.0 / phi[i][k] * z[k];
		}

        // update the phi
        for (::size_t k = 0; k < K; k++) {
			// FIXME RFHH a**0.5 * b**0.5 better written as sqrt(a*b) ?
            (*phi_star)[i][k] = std::abs(phi[i][k] + eps_t/2 * (alpha - phi[i][k] +
																grads[k]) +
										 std::pow(eps_t, .5) * std::pow(phi[i][k], .5) * noise[k]);
		}
	}


    void update_beta() {
		std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));
        // sums = np.sum(self.__theta,1)
		std::vector<double> sums(theta.size());
		std::transform(theta.begin(), theta.end(), sums.begin(), np::sum<double>);

        // update gamma, only update node in the grad
		double eps_t;
        if (! stepsize_switch) {
            eps_t = std::pow(1024+step_count, -0.5);
		} else {
            eps_t = a * std::pow(1 + step_count / b, -c);
		}

        for (::size_t i = 0; i < N; i++) {
            for (::size_t j = i + 1; j < N; j++) {
				Edge edge(i, j);

				if (edge.in(network.get_held_out_set()) || edge.in(network.get_test_set())) {
                    continue;
				}

                int y_ab = 0;
                if (edge.in(network.get_linked_edges())) {
                    y_ab = 1;
				}

                int z = sample_z_for_each_edge(y_ab, pi[i], pi[j],
											   beta, K);
                if (z == -1) {
                    continue;
				}

                grads[z][0] += std::abs(1-y_ab) / theta[z][0] - 1/ sums[z];
                grads[z][1] += std::abs(-y_ab) / theta[z][1] - 1 / sums[z];
			}
		}

        // update theta
		std::vector<std::vector<double> > noise = Random::random->randn(K, 2);
		std::vector<std::vector<double> > theta_star = np::clone(theta);
        for (::size_t k = 0; k < K; k++) {
            for (::size_t i = 0; i < 2; i++) {
				// FIXME rewrite a**0.5 * b**0.5 as sqrt(a * b)
				theta_star[k][i] = std::abs(theta[k][i] + eps_t/2 * (eta[i] - theta[k][i] + \
																	 grads[k][i]) +
										   	std::pow(eps_t, .5) * pow(theta[k][i], .5) * noise[k][i]);
			}
		}

        if (step_count < 50000) {
			// np::copy2D(&theta, theta_star);
			np::copy(&theta, theta_star);
		} else {
			// self.__theta = theta_star * 1.0/(self._step_count) + (1-1.0/(self._step_count))*self.__theta
			double inv_step_count = 1.0 / step_count;
			double one_inv_step_count = 1.0 - inv_step_count;
			MCMCMyOp<double> myOp(inv_step_count, one_inv_step_count);
			for (::size_t k = 0; k < theta.size(); k++) {
				std::transform(theta[k].begin(), theta[k].end(),
							   theta_star[k].begin(),
							   theta[k].begin(),
							   myOp);
			}
		}
        //self.__theta = theta_star
        // update beta from theta
        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
	}


    int sample_z_for_each_edge(double y, const std::vector<double> &pi_a, const std::vector<double> &pi_b, const std::vector<double> &beta, ::size_t K) const {
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
		std::vector<double> p(K + 1);
		// FIXME pow for selecting after y = 0 or 1
		for (::size_t k = 0; k < K; k++) {
            p[k] = std::pow(beta[k], y) * pow(1-beta[k], 1-y) * pi_a[k] * pi_b[k];
		}
        // p[K] = 1 - np.sum(p[0:K])
		p[K] = 0.0;
        p[K] = 1.0 - np::sum(p);

        // sample community based on probability distribution p.
        // bounds = np.cumsum(p)
		std::vector<double> bounds(K + 1);
		std::partial_sum(p.begin(), p.end(), bounds.begin());
        double location = Random::random->random() * bounds[K];

        // get the index of bounds that containing location.
        for (::size_t i = 0; i < K; i++) {
			if (location <= bounds[i]) {
				return i;
			}
		}

        return -1;
	}


	std::vector<int> sample_latent_vars(int node, const OrderedVertexSet &neighbor_nodes) const {
        /**
		 * given a node and its neighbors (either linked or non-linked), return the latent value
		 * z_ab for each pair (node, neighbor_nodes[i].
         */
		std::vector<int> z(K, 0);
        for (auto neighbor = neighbor_nodes.begin();
			 	neighbor != neighbor_nodes.end();
				neighbor++) {
            int y_ab = 0;      // observation
			Edge edge(std::min(node, *neighbor), std::max(node, *neighbor));
            if (edge.in(network.get_linked_edges())) {
                y_ab = 1;
			}

            int z_ab = sample_z_ab_from_edge(y_ab, pi[node], pi[*neighbor], beta, epsilon, K);
            z[z_ab]++;
		}

		return z;
	}


	/**
	 * run mini-batch based MCMC sampler
	 */
    virtual void run() {
        // pr = cProfile.Profile()
        // pr.enable()
        while (step_count < max_iteration && !is_converged()) {
            //print "step: " + str(self._step_count)
            double ppx_score = cal_perplexity_held_out();
			std::cout << std::fixed << std::setprecision(12) << "perplexity for held out set: " << ppx_score << std::endl;
            ppxs_held_out.push_back(ppx_score);

			std::vector<std::vector<double> > phi_star(pi);
            // iterate through each node, and update parameters pi_a
            for (::size_t i = 0; i < N; i++) {
                // update parameter for pi_i
                //print "updating: " + str(i)
                auto neighbor_nodes = sample_neighbor_nodes_batch(i);
				std::vector<int> z = sample_latent_vars(i, neighbor_nodes);
                update_pi_for_node(i, z, &phi_star, neighbor_nodes.size());
			}

			np::copy(&phi, phi_star);
			np::row_normalize(&pi, phi);

            // update beta
            update_beta();

            step_count++;
		}

#if 0
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
#endif
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

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
