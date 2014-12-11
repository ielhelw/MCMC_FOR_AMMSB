#ifndef MCMC_LEARNING_LEARNER_H__
#define MCMC_LEARNING_LEARNER_H__

#include <cmath>

#include "mcmc/types.h"
#include "mcmc/options.h"
#include "mcmc/network.h"

namespace mcmc {
namespace learning {

/**
 * This is base class for all concrete learners, including MCMC sampler, variational
 * inference,etc.
 */
class Learner {
public:
	/**
	 * initialize base learner parameters.
	 */
	Learner(const Options &args, const Network &network)
			: network(network) {

		// model priors
		alpha = args.alpha;
		eta.resize(2);
		eta[0] = args.eta0;
		eta[1] = args.eta1;
		average_count = 1;

		// parameters related to control model
		K = args.K;
		epsilon = args.epsilon;

		// parameters related to network
		N = network.get_num_nodes();

		// model parameters to learn
		beta = std::vector<double>(K, 0.0);
		pi   = std::vector<std::vector<double> >(N, std::vector<double>(K, 0.0));

		// parameters related to sampling
		mini_batch_size = args.mini_batch_size;
		if (mini_batch_size < 1) {
			mini_batch_size = N / 2;	// default option.
		}

		// ration between link edges and non-link edges
		link_ratio = network.get_num_linked_edges() / ((N * (N - 1)) / 2.0);
		// check the number of iterations.
		step_count = 1;
		// store perplexity for all the iterations
		// ppxs_held_out = [];
		// ppxs_test = [];

		max_iteration = args.max_iteration;
		CONVERGENCE_THRESHOLD = 0.000000000001;

		stepsize_switch = false;

		ppx_for_heldout = std::vector<double>(network.get_held_out_size());

		if (args.strategy == "unspecified") {
			strategy = strategy::STRATIFIED_RANDOM_NODE;
		} else if (args.strategy == "random-pair") {
			strategy = strategy::RANDOM_PAIR;
		} else if (args.strategy == "random-node") {
			strategy = strategy::RANDOM_NODE;
		} else if (args.strategy == "stratified-random-pair") {
			strategy = strategy::STRATIFIED_RANDOM_PAIR;
		} else if (args.strategy == "stratified-random-node") {
			strategy = strategy::STRATIFIED_RANDOM_NODE;
		} else {
			throw MCMCException("Unknown strategy type: " + args.strategy);
		}
	}

	virtual ~Learner() {
	}

	/**
	 * Each concrete learner should implement this. It basically
	 * iterate the data sets, then update the model parameters, until
	 * convergence. The convergence can be measured by perplexity score. 
	 *                                  
	 * We currently support four different learners:
	 * 1. MCMC for batch learning
	 * 2. MCMC for mini-batch training
	 * 3. Variational inference for batch learning
	 * 4. Stochastic variational inference
	 */
	virtual void run() = 0;

protected:
	void info(std::ostream &s) {
		s << "N " << N;
		s << " E " << network.get_num_total_edges();
	   	s << " K " << K;
		s << " iterations " << max_iteration;
		s << " minibatch size " << mini_batch_size;
		s << " link ratio " << link_ratio;
		s << " convergence " << CONVERGENCE_THRESHOLD;
		s << std::endl;
	}

	const std::vector<double> &get_ppxs_held_out() const {
		return ppxs_held_out;
	}

	const std::vector<double> &get_ppxs_test() const {
		return ppxs_test;
	}

	void set_max_iteration(::size_t max_iteration) {
		this->max_iteration = max_iteration;
	}

	double cal_perplexity_held_out() {
		return cal_perplexity(network.get_held_out_set());
	}

	double cal_perplexity_test() {
		return cal_perplexity(network.get_test_set());
	}

	bool is_converged() const {
		::size_t n = ppxs_held_out.size();
		if (n < 2) {
			return false;
		}
		if (std::abs(ppxs_held_out[n - 1] - ppxs_held_out[n - 2]) / ppxs_held_out[n - 2] >
				CONVERGENCE_THRESHOLD) {
			return false;
		}

		return true;
	}


protected:
	/**
	 * calculate the perplexity for data.
	 * perplexity defines as exponential of negative average log likelihood. 
	 * formally:
	 *     ppx = exp(-1/N * \sum){i}^{N}log p(y))
	 * 
	 * we calculate average log likelihood for link and non-link separately, with the 
	 * purpose of weighting each part proportionally. (the reason is that we sample 
	 * the equal number of link edges and non-link edges for held out data and test data,
	 * which is not true representation of actual data set, which is extremely sparse.
	 */
	double cal_perplexity(const EdgeMap &data) {
		double link_likelihood = 0.0;
		double non_link_likelihood = 0.0;
		::size_t link_count = 0;
		::size_t non_link_count = 0;

		::size_t i = 0;
		for (EdgeMap::const_iterator edge = data.begin();
			 	edge != data.end();
				edge++) {
			const Edge &e = edge->first;
			double edge_likelihood = cal_edge_likelihood(pi[e.first], pi[e.second],
														 edge->second, beta);
			if (std::isnan(edge_likelihood)) {
				std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
			}

			//cout<<"AVERAGE COUNT: " <<average_count;
			ppx_for_heldout[i] = (ppx_for_heldout[i] * (average_count-1) + edge_likelihood)/(average_count);
			// std::cerr << std::fixed << std::setprecision(12) << e << " in? " << (e.in(network.get_linked_edges()) ? "True" : "False") << " -> " << edge_likelihood << std::endl;
			// FIXME FIXME should not test again if we already know
			// assert(edge->second == e.in(network.get_linked_edges()));
			if (edge->second) {
				link_count++;
				link_likelihood += std::log(ppx_for_heldout[i]);
				//link_likelihood += edge_likelihood;

				if (std::isnan(link_likelihood)){
					std::cerr << "link_likelihood is NaN; potential bug" << std::endl;
				}
			} else {
				assert(! present(network.get_linked_edges(), e));
				non_link_count++;
				//non_link_likelihood += edge_likelihood;
				non_link_likelihood += std::log(ppx_for_heldout[i]);
				if (std::isnan(non_link_likelihood)){
					std::cerr << "non_link_likelihood is NaN; potential bug" << std::endl;
				}
			}
			i++;
		}
		// std::cerr << std::setprecision(12) << "ratio " << link_ratio << " count: link " << link_count << " " << link_likelihood << " non-link " << non_link_count << " " << non_link_likelihood << std::endl;

		// weight each part proportionally.
		// avg_likelihood = self._link_ratio*(link_likelihood/link_count) + \
		//         (1-self._link_ratio)*(non_link_likelihood/non_link_count)

		// direct calculation.
		double avg_likelihood = (link_likelihood + non_link_likelihood) / (link_count + non_link_count);
		if (true) {
			double avg_likelihood1 = link_ratio * (link_likelihood / link_count) + \
										 (1.0 - link_ratio) * (non_link_likelihood / non_link_count);
			std::cerr << std::fixed << std::setprecision(12) << avg_likelihood << " " << (link_likelihood / link_count) << " " << link_count << " " << \
				(non_link_likelihood / non_link_count) << " " << non_link_count << " " << avg_likelihood1 << std::endl;
			// std::cerr << "perplexity score is: " << exp(-avg_likelihood) << std::endl;
		}

		// return std::exp(-avg_likelihood);


		//if (step_count > 1000000)
		average_count = average_count + 1;
		std::cerr << "average_count is: " << average_count << " ";
		return (-avg_likelihood);
	}


	template <typename T>
	static void dump(const std::vector<T> &a, ::size_t n, const std::string &name = "") {
		n = std::min(n, a.size());
		std::cerr << name;
		if (n != a.size()) {
	   		std::cerr << "[0:" << n << "]";
		}
		std::cerr << " ";
		for (auto i = a.begin(); i < a.begin() + n; i++) {
			std::cerr << std::fixed << std::setprecision(12) << *i << " ";
		}
		std::cerr << std::endl;
	}


	/**
	 * calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
	 * in order to calculate this, we need to sum over all the possible (z_ab, z_ba)
	 * such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
	 * but this calculation can be done in O(K), by using some trick.
	 */
	double cal_edge_likelihood(const std::vector<double> &pi_a,
							   const std::vector<double> &pi_b,
							   bool y,
							   const std::vector<double> &beta) const {
		double s = 0.0;
#define DONT_FOLD_Y
#ifdef DONT_FOLD_Y
		if (y) {
			for (::size_t k = 0; k < K; k++) {
				s += pi_a[k] * pi_b[k] * beta[k];
			}
		} else {
			double sum = 0.0;
			for (::size_t k = 0; k < K; k++) {
				double f = pi_a[k] * pi_b[k];
				s += f * (1.0 - beta[k]);
				sum += f;
				assert(! std::isnan(f));
				assert(! std::isnan(s));
				assert(! std::isnan(sum));
			}
			s += (1.0 - sum) * (1.0 - epsilon);
		}
#else
		int iy = y ? 1 : 0;
		int y2_1 = 2 * iy - 1;
		int y_1 = iy - 1;
		double sum = 0.0;
		for (::size_t k = 0; k < K; k++) {
			double f = pi_a[k] * pi_b[k];
			sum += f;
			s += f * (beta[k] * y2_1 - y_1);
		}
		if (! y) {
			s += (1.0 - sum) * (1.0 - epsilon);
		}
#endif

		if (s < 1.0e-30) {
			s = 1.0e-30;
		}

		return s;
#if 0
		double prob = 0.0;
		double s = 0.0;

		for (::size_t k = 0; k < K; k++) {
			if (! y) {
				prob += pi_a[k] * pi_b[k] * (1 - beta[k]);
			} else {
				prob += pi_a[k] * pi_b[k] * beta[k];
			}
			s += pi_a[k] * pi_b[k];		// common expr w/ above
		}

		if (! y) {
			prob += (1.0 - s) * (1 - epsilon);
		} else {
			prob += (1.0 - s) * epsilon;
		}
		// std::cerr << "Calculate s " << s << " prob " << prob << std::endl;
		if (prob < 0.0) {
			std::cerr << "adsfadsfadsf" << std::endl;
		}

		return log(prob);
#endif
	}

protected:
	const Network &network;

	double alpha;
	std::vector<double> eta;
	::size_t K;
	double epsilon;
	::size_t N;

	std::vector<double> beta;
	std::vector<std::vector<double>> pi;

	::size_t mini_batch_size;
	double link_ratio;

	::size_t step_count;

	std::vector<double> ppxs_held_out;
	std::vector<double> ppxs_test;
	std::vector<double> iterations;
	std::vector<double> ppx_for_heldout;

	::size_t max_iteration;

	double CONVERGENCE_THRESHOLD;

	bool stepsize_switch;
	::size_t average_count;

	strategy::strategy strategy;
};


}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_LEARNER_H__
