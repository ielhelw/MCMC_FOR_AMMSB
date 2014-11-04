#ifndef MCMC_VARIATIONAL_INFERENCE_STOCHASTIC_H__
#define MCMC_VARIATIONAL_INFERENCE_STOCHASTIC_H__

#include <ctime>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <fstream>
#include <chrono>

#include <boost/math/special_functions/digamma.hpp>

#include "mcmc/np.h"

#include "mcmc/types.h"
#include "mcmc/data.h"
#include "mcmc/random.h"
#include "mcmc/estimate_phi.h"
#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

typedef std::unordered_map<Edge, std::vector<double> >	PhiMap;


/**
 *  Stochastic variational inference for assortive mixture membership stochastic model.
 *  The implementation is based on the paper:
 *      http://www.cs.princeton.edu/~blei/papers/GopalanMimnoGerrishFreedmanBlei2012.pdf
 *
 *  Formally, each node can be belong to multiple communities which we can represent it by
 *  distribution of communities. For instance, if we assume there are total K communities
 *  in the graph, then each node a, is attached to community distribution \pi_{a}, where
 *  \pi{a} is K dimensional vector, and \pi_{ai} represents the probability that node a
 *  belongs to community i, and \pi_{a0} + \pi_{a1} +... +\pi_{aK} = 1
 *
 *  Also, there is another parameters called \beta representing the community strength, where
 *  \beta_{k} is scalar.
 *
 *  In summary, the model has the parameters:
 *  Prior: \alpha, \eta
 *  Parameters: \pi, \beta
 *  Latent variables: z_ab, z_ba
 *  Observations: y_ab for every link.
 *
 *  And our goal is to estimate the posterior given observations and priors:
 *  p(\pi,\beta | \alpha,\eta, y).
 *
 *  Due to the intractability of this posterior, we adopt approximate inference - variational inference
 *  More specifically, using the mean-field variational inference.
 *
 *  In this implementation, we introduce sets of variational parameters.
 *  q(z_ab) ~ Mul(phi_ab)     phi_ab is K dimensional vector, where sum equals to 1
 *  q(z_ba) ~ Mul(phi_ba)     phi_ba is K dimensional vector, where sum equals to 1
 *  q(pi_a) ~ Dir(gamma_a)    gamma_a is K dimensional vector,  for each node a.
 *  q(beta_k) ~ Beta(lamda_k) lamda_k is 2 dimensional vector, each denotes the beta shape parameters.
 *
 *  TODO:  need to create base class for sampler, and MCMC sampler and variational inference should inherit
 *         from that base class.
 */
class SVI : public Learner {

public:
	/**
	 * Initialize the sampler using the network object and arguments (i.e prior)
	 * Arguments:
	 * 	   network:    representation of the graph.
	 * 	   args:       containing priors, control parameters for the model.
	 */
	SVI(const Options &args, const Network &graph)
   			: Learner(args, graph) {
		// variational parameters.
		lamda = Random::random->gamma(eta[0], eta[1], K, 2);	// variational parameters for beta
		gamma = Random::random->gamma(1, 1, N, K);			// variational parameters for pi
		std::cerr << "gamma.size() " << gamma.size() << " gamma[0].size() " << gamma[0].size() << std::endl;
		update_pi_beta();
		// step size parameters.
		kappa = args.b;
		tao = args.c;

		// control parameters for learning
		online_iterations = 50;
		phi_update_threshold = 0.0001;

		// lift
		log_epsilon = log(epsilon);
		log_1_epsilon = log(1.0 - epsilon);

		info(std::cout);
	}


	virtual ~SVI() {
	}


	virtual void run() {
		/*
		   stochastic variational optimization.
		   while not converge
		   sample mini-batch node pairs E_t  from network (trianing)
		   for each node pair (a,b) in E_t
		   optimize (phi_ab, phi_ba)
		   for n = [0,..,N-1], k=[0,..K-1]
		   calculate the gradient for gamma_nk
		   for k = [0,,,K-1], i=[0,1]
		   calculate the gradient for lammda_ki
		   update (gamma, lamda) using gradient:
		   new_value = (1-p_t)*old_value + p_t * new value.
		 */

		// pr = cProfile.Profile()
		// pr.enable()

		// running until convergence.
		time_t start = time(NULL);
		step_count++;

		while (step_count < max_iteration and !is_converged()) {
			auto l1 = std::chrono::system_clock::now();
			// (mini_batch, scale) = network.sample_mini_batch(mini_batch_size, "stratified-random-node")
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			/*
			pr = cRpofile.Profile()
			pr.enable()
			*/
			// evaluate model after processing every 10 mini-batches.
			if (step_count % 1 == 0) {
				double ppx_score = cal_perplexity_held_out();
				std::cout << "perplexity for hold out set is: " << std::setprecision(15) << ppx_score << std::endl;
				ppxs_held_out.push_back(ppx_score);

                if (step_count > 5000) {
                    ::size_t size = avg_log.size();
                    ppx_score = (1-1.0/(step_count-50)) * avg_log[size-1] + 1.0/(step_count-50) * ppx_score;
                    avg_log.push_back(ppx_score);
				} else {
                    avg_log.push_back(ppx_score);
				}

                timing.push_back(time(NULL)-start);
			}

            if (step_count % 50 == 0) {
                this->save();
			}

			// update (phi_ab, phi_ba) for each edge
			PhiMap phi;	// mapping (a,b) => (phi_ab, phi_ba)
			sample_latent_vars_for_edges(&phi, mini_batch);
			update_gamma_and_lamda(phi, mini_batch, scale);
			// std::cerr << "old beta: " << std::setprecision(15) << beta;
			update_pi_beta();
			// std::cerr << "new beta: " << std::setprecision(15) << beta;

			step_count++;
			auto l2 = std::chrono::system_clock::now();
			std::cout << "LOOP  = " << (l2-l1).count() << std::endl;
		}

#if 0
		pr.disable()
			s = StringIO.StringIO()
			sortby = 'cumulative'
			ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
			ps.print_stats()
			print s.getvalue()
#else
		std::cerr << "Would like to print stats here" << std::endl;
#endif
	}


protected:
	void sample_latent_vars_for_edges(PhiMap *phi, const OrderedEdgeSet &mini_batch) const {
		if (false) {
			std::cerr << "Minibatch size " << mini_batch.size() << std::endl;
		}
		for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
			int a = edge->first;
			int b = edge->second;
			//estimate_phi_for_edge(edge, phi)  // this can be done in parallel.

			if (false) {
				std::cerr << "Investigate " << *edge << std::endl;
			}
			sample_latent_vars_for_each_pair(a, b, gamma[a], gamma[b],
											 lamda, K, phi_update_threshold,
											 epsilon, online_iterations,
											 network.get_linked_edges(),
											 &(*phi)[Edge(a,b)],
											 &(*phi)[Edge(b,a)]);
			// std::cerr << "phi[(a,b)] " << (*phi)[Edge(a,b)] << std::endl;
			// std::cerr << "phi[(b,a)] " << (*phi)[Edge(b,a)] << std::endl;

			//estimate_phi_for_edge(edge, phi)
		}
	}


	void update_pi_beta() {
		/**
		 * Spell this out from numpy terms
		 * gamma :: double[N,K]
		 *	pi = gamma/np.sum(gamma,1)[:,np.newaxis]
		 *		s = np.sum(gamma,1) :: double[N]; s[i] = sum_j gamma[i,j]
		 *		s = map(sum(gamma))
		 *		t = s[:,np.newaxis] :: double[N,1]; t[i,0] = s[i]
		 *		pi = gamma/t :: double[N,K]; pi[i,j] = gamma[i,j] / t[i,0] = gamma[i,j] / s[i]
		 *
		 * lamda :: double[K,2]
		 * temp = lamda/np.sum(lamda,1)[:,np.newaxis]
		 * beta = temp[:,1]
		 * 		s = np.sum(lamda,1) :: double[K]; s[i] sum_j lamda[i,j]
		 * 		temp = gamma/t :: double[K,2]; temp[i,j] = lamda[i,j] / s[i]
		 */
		// pi = gamma/np.sum(gamma,1)[:,np.newaxis];
		np::row_normalize(&pi, gamma);

		// temp = lamda/np.sum(lamda,1)[:,np.newaxis];
		// beta = temp[:,1];
		std::vector<std::vector<double> > temp(lamda.size(), std::vector<double>(lamda[0].size()));
		np::row_normalize(&temp, lamda);
		// std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(0));
	}


	void update_gamma_and_lamda(PhiMap &phi, const OrderedEdgeSet &mini_batch, double scale) {

		// calculate the gradient for gamma
		std::vector<std::vector<double> > grad_lamda(K, std::vector<double>(2, 0.0));
		std::unordered_map<int, std::vector<double> > grad_gamma(N);	// ie. grad[a] = array[] which is K dimensional vector
		std::unordered_map<int, ::size_t> counter;	// used for scaling
		for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
			/*
			 * calculate the gradient for gamma
			 */
			int a = edge->first;
			int b = edge->second;
			const std::vector<double> &phi_ab = phi[Edge(a,b)];
			const std::vector<double> &phi_ba = phi[Edge(b,a)];
			if (grad_gamma.find(a) != grad_gamma.end()) {
				std::transform(grad_gamma[a].begin(), grad_gamma[a].end(),
							   phi_ab.begin(),
							   grad_gamma[a].begin(),
							   std::plus<double>());
				counter[a]++;
			} else {
				grad_gamma[a] = phi_ab;
				counter[b] = 1;
			}

			if (grad_gamma.find(b) != grad_gamma.end()) {
				std::transform(grad_gamma[b].begin(), grad_gamma[b].end(),
							   phi_ba.begin(),
							   grad_gamma[b].begin(),
							   std::plus<double>());
				counter[b]++;
			} else {
				grad_gamma[b] = phi_ba;
				counter[b] = 1;
			}

			/*
			 * calculate the gradient for lambda
			 */
			int y = 0;
			if (edge->in(network.get_linked_edges())) {
				y = 1;
			}

			for (::size_t k = 0; k < K; k++) {
				grad_lamda[k][0] += phi_ab[k] * phi_ba[k] * y;
				grad_lamda[k][1] += phi_ab[k] * phi_ba[k] * (1-y);
			}
		}


		// update gamma, only update node in the grad
		double p_t;
		if (! stepsize_switch) {
			p_t = std::pow(1024 + step_count, -0.5);
		} else {
			p_t = 0.01* std::pow(1+step_count/1024.0, -0.55);
		}

		struct MyOpIn {
			MyOpIn(double p_t, double alpha, double scale) : p(1.0 - p_t), a(p_t * alpha), s(p_t * scale) {
			}

			double operator() (const double &x, const double &y) {
				return p * x + a + s * y;
				// p_t * (alpha + scale * y);
			}

			double p;
			double a;
			double s;
		};
		struct MyOpNotIn {
			MyOpNotIn(double p_t, double alpha) : p(1.0 - p_t), a(p_t * alpha) {
			}

			double operator() (const double &x) {
				return p * x + a;
			}

			double p;
			double a;
		};
		MyOpIn myOpIn(p_t, alpha, scale);
		MyOpNotIn myOpNotIn(p_t, alpha);
		for (::size_t node = 0; node < N; node++) {
			if (grad_gamma.find(node) != grad_gamma.end()) {
				std::transform(gamma[node].begin(), gamma[node].end(), grad_gamma[node].begin(),
							   gamma[node].begin(), myOpIn);
			} else {
				std::transform(gamma[node].begin(), gamma[node].end(),
							   gamma[node].begin(), myOpNotIn);
			}
		}
#if 0
		for (std::unordered_map<int, std::vector<double> >::iterator node = grad_gamma.begin();
			 	node != grad_gamma.end();
				node++) {
			if (counter[node->first] == 0) {
				continue;
			}

			// std::vector<double> gamma_star(K);
			double scale1 = 1.0;
			if (! flag) {
				scale1 = N/(counter[node->first]*1.0);
			}

			if (step_count > 400) {
				for (::size_t k = 0; k < K; k++) {
					// gamma_star[k] = (1-p_t)*gamma[node->first][k] + p_t * (alpha[k] + scale1 * grad_gamma[node->first][k]);
					// gamma[node->first][k] = (1-1.0/(step_count))*gamma[node->first][k] + 1.0/(step_count)*gamma_star[k];
					double gamma_star = (1-p_t)*gamma[node->first][k] + p_t * (alpha + scale1 * grad_gamma[node->first][k]);
					gamma[node->first][k] = (1-1.0/(step_count))*gamma[node->first][k] + 1.0/(step_count)*gamma_star;
				}
			} else {
				for (::size_t k = 0; k < K; k++) {
					gamma[node->first][k]=(1-p_t)*gamma[node->first][k] + p_t * (alpha + scale1 * grad_gamma[node->first][k]);
				}
			}
		}
#endif

		// update lamda
		// std::cerr << std::setprecision(17) << "p_t " << p_t << " eta (" << eta[0] << "," << eta[1] << ")" << " scale " << scale << " step_count " << step_count << std::endl;
		for (::size_t k = 0; k < K; k++) {

			if (step_count > 400000) {
				double lamda_star_0 = (1-p_t)*lamda[k][0] + p_t *(eta[0] + scale * grad_lamda[k][0]);
				double lamda_star_1 = (1-p_t)*lamda[k][1] + p_t *(eta[1] + scale * grad_lamda[k][1]);
				lamda[k][0] = (1-1.0/(step_count)) * lamda[k][0] +1.0/(step_count)*lamda_star_0;
				lamda[k][1] = (1-1.0/(step_count)) * lamda[k][1] +1.0/(step_count)*lamda_star_1;
			} else {
				lamda[k][0] = (1-p_t)*lamda[k][0] + p_t *(eta[0] + scale * grad_lamda[k][0]);
				lamda[k][1] = (1-p_t)*lamda[k][1] + p_t *(eta[1] + scale * grad_lamda[k][1]);
			}
		}
	}


#ifdef UNUSED
	void estimate_phi_for_edge(const Edge &edge, PhiMap *phi) {

		/**
		calculate (phi_ab, phi_ba) for given edge : (a,b)
		(a) calculate phi_ab given phi_ba
			if y =0:
		phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(1-epsilon))
			if y=1:
		phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(epsilon))

		(b) calculate phi_ba given phi_ab
			if y =0:
		phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(1-epsilon))
			if y=1:
		phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(epsilon))

		*/

		using ::boost::math::digamma;

		int a = edge.first;
		int b = edge.second;
		// initialize
		std::vector<double> phi_ab(K, 1.0 / K);
		std::vector<double> phi_ba(K, 1.0 / K);

		bool y = false;
		if (edge.in(network.get_linked_edges())) {
			y = true;
		}

		// alternatively update phi_ab and phi_ba, until it converges
		// or reach the maximum iterations.
		for (::size_t i = 0; i < online_iterations; i++) {
			std::vector<double> phi_ab_old(phi_ab);
			std::vector<double> phi_ba_old(phi_ba);

			// first, update phi_ab
			for (::size_t k = 0; k < (K); k++) {
				if (y) {
					double u = -phi_ba[k]* log_epsilon;
					phi_ab[k] = std::exp(digamma(gamma[a][k])+phi_ba[k]*\
										 (digamma(lamda[k][0])-digamma(lamda[k][0]+lamda[k][1]))+u);
				} else {
					double u = -phi_ba[k]* log_1_epsilon;
					phi_ab[k] = std::exp(digamma(gamma[a][k])+phi_ba[k]*\
										 (digamma(lamda[k][1])-digamma(lamda[k][0]+lamda[k][1]))+u);
				}
			}
			double sum_phi_ab = np::sum(phi_ab);
			for (::size_t k = 0; k < phi_ab.size(); k++) {
				phi_ab[k] /= sum_phi_ab;
			}

			// then update phi_ba
			for (::size_t k = 0; k < K; k++) {
				if (y) {
					double u = -phi_ab[k]* log_epsilon;
					phi_ba[k] = std::exp(digamma(gamma[b][k])+phi_ab[k]*
										 (digamma(lamda[k][0])-digamma(lamda[k][0]+lamda[k][1]))+u);
				} else {
					double u = -phi_ab[k]* log_1_epsilon;
					phi_ba[k] = std::exp(digamma(gamma[b][k])+phi_ab[k]*
										 (digamma(lamda[k][1])-digamma(lamda[k][0]+lamda[k][1]))+u);
				}
			}

			double sum_phi_ba = np::sum(phi_ba);
			for (::size_t k = 0; k < phi_ba.size(); k++) {
				phi_ba[k] /= sum_phi_ba;
			}

			// calculate the absolute difference between new value and old value
			// diff1 = np.sum(np.abs(phi_ab - phi_ab_old))
			// diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
			double diff1 = np::sum_abs(phi_ab, phi_ab_old);
			double diff2 = np::sum_abs(phi_ba, phi_ba_old);
			if (diff1 < phi_update_threshold and diff2 < phi_update_threshold) {
				break;
			}
		}

		(*phi)[Edge(a,b)] = phi_ab;
		(*phi)[Edge(b,a)] = phi_ba;
	}
#endif
    
    
    void save() {
#if 0
		std::ofstream f;
        f.open('ppx_variational_sampler.txt');
        for (::size_t i = 0; i < avg_log.size(); i++) {
            f << std::exp(avg_log[i]) << "\t" << timing[i] << std::endl;
		}
        f.close();
#else
		std::cerr << "Would like to dump avg_log and timing" << std::endl;
#endif
	}


protected:
	// replicated in both variational_inference_*
	std::vector<std::vector<double> > lamda;	// variational parameters for beta
	std::vector<std::vector<double> > gamma;	// variational parameters for pi
	double kappa;
	double tao;

	// control parameters for learning
	::size_t online_iterations;
	double phi_update_threshold;

	double log_epsilon;
	double log_1_epsilon;

	std::vector<double> avg_log;
	std::vector<time_t> timing;
};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_VARIATIONAL_INFERENCE_STOCHASTIC_H__
