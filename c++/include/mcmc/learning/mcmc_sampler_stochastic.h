#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"

#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

#ifdef UNUSED
#ifdef RANDOM_FOLLOWS_CPP
#define EDGEMAP_IS_VECTOR
#endif

// EDGEMAP_IS_VECTOR is a more efficient implementation anyway
#ifdef EDGEMAP_IS_VECTOR
typedef std::vector<int>    EdgeMapZ;
#else
// typedef std::map<Edge, int>	EdgeMapZ;
typedef std::unordered_map<Edge, int>   EdgeMapZ;
#endif
#endif

#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
#  define NEIGHBOR_SET_IS_VECTOR
typedef std::vector<int> NeighborSet;
#else
typedef OrderedVertexSet NeighborSet;
#endif


class MCMCSamplerStochastic : public Learner {
public:
    /**
    Mini-batch based MCMC sampler for community overlapping problems. Basically, given a
    connected graph where each node connects to other nodes, we try to find out the
    community information for each node.

    Formally, each node can be belong to multiple communities which we can represent it by
    distribution of communities. For instance, if we assume there are total K communities
    in the graph, then each node a, is attached to community distribution \pi_{a}, where
    \pi{a} is K dimensional vector, and \pi_{ai} represents the probability that node a
    belongs to community i, and \pi_{a0} + \pi_{a1} +... +\pi_{aK} = 1

    Also, there is another parameters called \beta representing the community strength, where
    \beta_{k} is scalar.

    In summary, the model has the parameters:
    Prior: \alpha, \eta
    Parameters: \pi, \beta
    Latent variables: z_ab, z_ba
    Observations: y_ab for every link.

    And our goal is to estimate the posterior given observations and priors:
    p(\pi,\beta | \alpha,\eta, y).

    Because of the intractability, we use MCMC(unbiased) to do approximate inference. But
    different from classical MCMC approach, where we use ALL the examples to update the
    parameters for each iteration, here we only use mini-batch (subset) of the examples.
    This method is great marriage between MCMC and stochastic methods.
    */
    MCMCSamplerStochastic(const Options &args, const Network &graph)
			: Learner(args, graph),
			args(args),
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
			kernelRandom(Random::random)
#else
			kernelRandom(new Random::Random(42))
#endif
	{
#ifdef RANDOM_FOLLOWS_CPP
		std::cerr << "RANDOM_FOLLOWS_CPP enabled" << std::endl;
#endif
#ifdef EFFICIENCY_FOLLOWS_CPP_WENZHE
		std::cerr << "EFFICIENCY_FOLLOWS_CPP_WENZHE enabled" << std::endl;
#  ifndef RANDOM_FOLLOWS_CPP_WENZHE
#    define RANDOM_FOLLOWS_CPP_WENZHE
#  endif
#endif
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
		std::cerr << "RANDOM_FOLLOWS_CPP_WENZHE enabled" << std::endl;
#endif
#ifdef RANDOM_SYSTEM
		std::cerr << "RANDOM_SYSTEM enabled" << std::endl;
#endif

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning
        //num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
		if (args.num_node_sample == 0) {
			// TODO: automative update..... 
			num_node_sample = N/50;
		} else {
			num_node_sample = args.num_node_sample;
		}
		if (args.interval == 0) {
			interval = 50;
		} else {
			interval = args.interval;
		}
		if (args.mini_batch_size == 0) {
			mini_batch_size = N / 10;   // old default for STRATIFIED_RANDOM_NODE_SAMPLING
		}
		std::cerr << "num_node_sample " << num_node_sample << " a " << a << " b " << b << " c " << c << " alpha " << alpha << " eta (" << eta[0] << "," << eta[1] << ")" << std::endl;

		info(std::cout);
	}

	virtual void init() {
        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		theta = kernelRandom->gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
		// std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
		// theta = kernelRandom->gamma(100.0, 0.01, K, 2);		// parameterization for \beta

		// FIXME RFHH -- code sharing with variational_inf*::update_pi_beta()
        // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));

		phi = kernelRandom->gamma(1, 1, N, K);					// parameterization for \pi
		std::cerr << "Done host random for phi" << std::endl;
#ifndef NDEBUG
        for (auto pph : phi) {
          for (auto ph : pph) {
            assert(ph >= 0.0);
          }
        }
#endif
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);

		if (true) {
			std::cout << std::fixed << std::setprecision(12) << "beta[0] " << beta[0] << std::endl;
		} else {
			std::cerr << "beta ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << beta[k] << " ";
			}
			std::cerr << std::endl;
		}

		if (false) {
			std::cout << "theta[*][0]: ";
			for (::size_t k = 0; k < K; k++) {
				std::cout << std::fixed << std::setprecision(12) << theta[k][0] << " ";
			}
			std::cout << std::endl;
			std::cout << "theta[*][1]: ";
			for (::size_t k = 0; k < K; k++) {
				std::cout << std::fixed << std::setprecision(12) << theta[k][1] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "phi[0][0] " << phi[0][0] << std::endl;
		if (false) {
			std::cout << "pi[0] ";
			for (::size_t k = 0; k < K; k++) {
				std::cout << pi[0][k] << " ";
			}
			std::cout << std::endl;
		}

		if (true) {
			for (::size_t i = 0; i < 10; i++) {
				std::cerr << "phi[" << i << "]: ";
				for (::size_t k = 0; k < 10; k++) {
					std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
				}
				std::cerr << std::endl;
				std::cerr << "pi[" << i << "]: ";
				for (::size_t k = 0; k < 10; k++) {
					std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
				}
				std::cerr << std::endl;
			}
		}

        std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
		std::cerr << "Done constructor" << std::endl;
	}

	virtual ~MCMCSamplerStochastic() {
#ifndef RANDOM_FOLLOWS_CPP_WENZHE
		delete kernelRandom;
#endif
	}

#if 0
    def run1(self):
        while self._step_count < self._max_iteration and not self._is_converged():
            /**
            pr = cProfile.Profile()
            pr.enable()
             */
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            //print "iteration: " + str(self._step_count)

            if self._step_count % 1 == 0:
				#print str(self._beta)
                ppx_score = self._cal_perplexity_held_out()
				#print "perplexity for hold out set is: "  + str(ppx_score)
                self._ppxs_held_out.append(ppx_score)

            self.__update_pi1(mini_batch, scale)

            // sample (z_ab, z_ba) for each edge in the mini_batch.
            // z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale,z)

            /**
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
             */
            self._step_count += 1

        print "terminated"
#endif

    virtual void run() {
        /** run mini-batch based MCMC sampler, based on the sungjin's note */
		timer::Timer t_outer("  outer");
		timer::Timer t_perplexity("  perplexity");
		timer::Timer t_mini_batch("  sample_mini_batch");
		timer::Timer t_nodes_in_mini_batch("  nodes_in_mini_batch");
		timer::Timer t_sample_neighbor_nodes("  sample_neighbor_nodes");
		timer::Timer t_update_phi("  update_phi");
		timer::Timer t_update_pi("  update_pi");
		timer::Timer t_update_beta("  update_beta");
		timer::Timer::setTabular(true);

        using namespace std::chrono;

		clock_t t1, t2;
		std::vector<double> timings;
		t1 = clock();
        while (step_count < max_iteration && ! is_converged()) {
			t_outer.start();
			auto l1 = std::chrono::system_clock::now();
			//if (step_count > 200000){
				//interval = 2;
			//}
			if (step_count % interval == 0) {
				t_perplexity.start();
				double ppx_score = cal_perplexity_held_out();
				t_perplexity.stop();
				std::cout << std::fixed << std::setprecision(12) << "step count: " << step_count << " perplexity for hold out set: " << ppx_score << std::endl;
				ppxs_held_out.push_back(ppx_score);

				t2 = clock();
				double diff = (double)t2 - (double)t1;
				double seconds = diff / CLOCKS_PER_SEC;
				timings.push_back(seconds);
				iterations.push_back(step_count);
#if 0
				if (ppx_score < 5.0) {
					stepsize_switch = true;
					//print "switching to smaller step size mode!"
				}
#endif
			}

			// write into file
			if (step_count % 2000 == 1) {
				if (false) {
					std::ofstream myfile;
					std::string file_name = "mcmc_stochastic_" + std::to_string (K) + "_num_nodes_" + std::to_string(num_node_sample) + "_us_air.txt";
					myfile.open (file_name);
					int size = ppxs_held_out.size();
					for (int i = 0; i < size; i++){

						//int iteration = i * 100 + 1;
						myfile <<iterations[i]<<"    "<<timings[i]<<"    "<<ppxs_held_out[i]<<"\n";
					}

					myfile.close();
				}
			}

            //print "step: " + str(self._step_count)
            /**
            pr = cProfile.Profile()
            pr.enable()
             */

            // (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			// std::cerr << "Invoke sample_mini_batch" << std::endl;
			t_mini_batch.start();
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
			t_mini_batch.stop();
			if (false) {
				std::cerr << "Minibatch: ";
				for (auto e : *edgeSample.first) {
					std::cerr << e << " ";
				}
				std::cerr << std::endl;
			}
			// std::cerr << "Done sample_mini_batch" << std::endl;
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			//std::unordered_map<int, std::vector<int> > latent_vars;
			//std::unordered_map<int, ::size_t> size;

            // iterate through each node in the mini batch.
			t_nodes_in_mini_batch.start();
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();
// std::cerr << "mini_batch size " << mini_batch.size() << " num_node_sample " << num_node_sample << std::endl;

#ifndef EFFICIENCY_FOLLOWS_CPP_WENZHE
			double eps_t  = a * std::pow(1 + step_count / b, -c);	// step size
			// double eps_t = std::pow(1024+step_count, -0.5);
#endif

			// ************ do in parallel at each host
			// std::cerr << "Sample neighbor nodes" << std::endl;
			// FIXME: nodes_in_batch should generate a vector, not an OrderedVertexSet
			std::vector<int> node_vector(nodes.begin(), nodes.end());
#pragma omp parallel for
			for (::size_t n = 0; n < node_vector.size(); ++n) {
				int node = node_vector[n];
				t_sample_neighbor_nodes.start();
				// sample a mini-batch of neighbors
				NeighborSet neighbors = sample_neighbor_nodes(num_node_sample, node);
				t_sample_neighbor_nodes.stop();

                // std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
				t_update_phi.start();
				update_phi(node, neighbors
#ifndef EFFICIENCY_FOLLOWS_CPP_WENZHE
						   , eps_t
#endif
						   );
				t_update_phi.stop();
			}

			// ************ do in parallel at each host
			t_update_pi.start();
#if defined EFFICIENCY_FOLLOWS_CPP_WENZHE
			// std::cerr << __func__ << ":" << __LINE__ << ":  FIXME" << std::endl;
			np::row_normalize(&pi, phi);	// update pi from phi.
#else
			// No need to update pi where phi is unchanged
			for (auto i: nodes) {
				np::normalize(&pi[i], phi[i]);
			}
#endif
			t_update_pi.stop();

			t_update_beta.start();
            update_beta(mini_batch, scale);
			t_update_beta.stop();

			delete edgeSample.first;

            step_count++;
			t_outer.stop();
			auto l2 = std::chrono::system_clock::now();
			if (false) {
				std::cout << "LOOP  = " << (l2-l1).count() << std::endl;
			}
		}

		timer::Timer::printHeader(std::cout);
		std::cout << t_outer << std::endl;
		std::cout << t_perplexity << std::endl;
		std::cout << t_mini_batch << std::endl;
		std::cout << t_nodes_in_mini_batch << std::endl;
		std::cout << t_sample_neighbor_nodes << std::endl;
		std::cout << t_update_phi << std::endl;
		std::cout << t_update_pi << std::endl;
		std::cout << t_update_beta << std::endl;
	}


protected:

#if 0
    def __update_pi1(self, mini_batch, scale):

        grads = np.zeros((self._N, self._K))
        counter = np.zeros(self._N)
        phi_star = np.zeros((self._N, self._K))

        for edge in mini_batch:
            a = edge[0]
            b = edge[1]

            y_ab = 0      // observation
            if (min(a, b), max(a, b)) in self._network.get_linked_edges():
                y_ab = 1

            z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[a], self._pi[b], self._beta, self._epsilon, self._K)
            z_ba = self.sample_z_ab_from_edge(y_ab, self._pi[b], self._pi[a], self._beta, self._epsilon, self._K)


            counter[a] += 1
            counter[b] += 1

            grads[a][z_ab] += 1/self.__phi[a][z_ab]
            grads[b][z_ba] += 1/self.__phi[b][z_ba]

         // update gamma, only update node in the grad
        if self.stepsize_switch == False:
            eps_t = (1024+self._step_count)**(-0.5)
        else:
            eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)

        for i in range(0, self._N):
            noise = random.randn(self._K)
            sum_phi_i = np.sum(self.__phi[i])
            for k in range(0, self._K):

                if counter[i] < 1:
                    phi_star[i][k] = abs((self.__phi[i,k]) + eps_t*(self._alpha - self.__phi[i,k])+(2*eps_t)**.5*self.__phi[i,k]**.5 * noise[k])
                else:
                    phi_star[i][k] = abs(self.__phi[i,k] + eps_t * (self._alpha - self.__phi[i,k] + \
                                scale * (grads[i][k]-(1.0/sum_phi_i)*counter[i])) \
                                + (2*eps_t)**.5*self.__phi[i,k]**.5 * noise[k])

                if self._step_count < 50000:
                    self.__phi[i][k] = phi_star[i][k]
                else:
                    self.__phi[i][k] = phi_star[i][k] * (1.0/(self._step_count)) + \
                                                (1-(1.0/(self._step_count)))*self.__phi[i][k]

            sum_phi = np.sum(self.__phi[i])
            self._pi[i] = [self.__phi[i,k]/sum_phi for k in range(0, self._K)]
#endif


    void update_beta(const OrderedEdgeSet &mini_batch, double scale) {

		std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));	// gradients K*2 dimension
		std::vector<double> probs(K);
        // sums = np.sum(self.__theta,1)
		std::vector<double> theta_sum(theta.size());
		std::transform(theta.begin(), theta.end(), theta_sum.begin(), np::sum<double>);

		// update gamma, only update node in the grad
		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		//double eps_t = std::pow(1024+step_count, -0.5);
		for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
            int y = 0;
            if (edge->in(network.get_linked_edges())) {
                y = 1;
			}
			int i = edge->first;
			int j = edge->second;

			double pi_sum = 0.0;
			for (::size_t k = 0; k < K; k++) {
				pi_sum += pi[i][k] * pi[j][k];
#ifdef EFFICIENCY_FOLLOWS_CPP_WENZHE
				probs[k] = std::pow(beta[k], y) * std::pow(1 - beta[k], 1 - y) * pi[i][k] * pi[j][k];
#else
				double f = pi[i][k] * pi[j][k];
				if (y == 1) {
					probs[k] = beta[k] * f;
				} else {
					probs[k] = (1.0 - beta[k]) * f;
				}
#endif
			}

#if defined EFFICIENCY_FOLLOWS_CPP_WENZHE
			double prob_0 = std::pow(epsilon, y) * std::pow(1 - epsilon, 1 - y) * (1 - pi_sum);
			double prob_sum = np::sum(probs) + prob_0;
			for (::size_t k = 0; k < K; k++) {
				grads[k][0] += (probs[k] / prob_sum) * (std::abs(1 - y) / theta[k][0] - 1 / theta_sum[k]);
				grads[k][1] += (probs[k] / prob_sum) * (std::abs(-y) / theta[k][1] - 1 / theta_sum[k]);
			}
#else
			double prob_0 = ((y == 1) ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
			double prob_sum = np::sum(probs) + prob_0;
			for (::size_t k = 0; k < K; k++) {
				double f = probs[k] / prob_sum;
				double one_over_theta_sum = 1.0 / theta_sum[k];
				grads[k][0] += f * ((1 - y) / theta[k][0] - one_over_theta_sum);
				grads[k][1] += f * (y / theta[k][1] - one_over_theta_sum);
			}
#endif
		}

        // update theta
		std::vector<std::vector<double> > noise = kernelRandom->randn(K, 2);	// random noise.
		// std::vector<std::vector<double> > theta_star(theta);
        for (::size_t k = 0; k < K; k++) {
            for (::size_t i = 0; i < 2; i++) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
				// FIXME rewrite a**0.5 * b**0.5 as sqrt(a * b)
				theta[k][i] = std::abs(theta[k][i] + eps_t / 2 * (eta[i] - theta[k][i] + \
																  scale * grads[k][i]) +
									   std::pow(eps_t, .5) * std::pow(theta[k][i], .5) * noise[k][i]);
#else
				double f = std::sqrt(eps_t * theta[k][i]);
				theta[k][i] = std::abs(theta[k][i] + eps_t / 2.0 * (eta[i] - theta[k][i] + \
																	scale * grads[k][i]) +
									   f * noise[k][i]);
#endif
			}
		}

		// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
		// self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));

        if (false) {
          for (auto n : noise) {
            std::cerr << "noise ";
            for (auto b : n) {
              std::cerr << std::fixed << std::setprecision(12) << b << " ";
            }  
            std::cerr << std::endl;
          }
          std::cerr << "beta ";
          for (auto b : beta) {
            std::cerr << std::fixed << std::setprecision(12) << b << " ";
          }
          std::cerr << std::endl;
        }
	}


    void update_phi(int i, const NeighborSet &neighbors
#ifndef EFFICIENCY_FOLLOWS_CPP_WENZHE
						   , double eps_t
#endif
					) {
#ifdef EFFICIENCY_FOLLOWS_CPP_WENZHE
		double eps_t  = a * std::pow(1 + step_count / b, -c);	// step size
		// double eps_t = std::pow(1024+step_count, -0.5);
#endif

		if (false) {
			std::cerr << "update_phi pre phi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
			}
			std::cerr << std::endl;
			std::cerr << "pi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
			}
			std::cerr << std::endl;
			for (auto n: neighbors) {
				std::cerr << "pi[" << n << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << pi[n][k] << " ";
				}
				std::cerr << std::endl;
			}
		}

		double phi_i_sum = np::sum(phi[i]);
        std::vector<double> grads(K, 0.0);	// gradient for K classes
		// std::vector<double> phi_star(K);					// temp vars

		for (auto neighbor: neighbors) {
			if (i == neighbor) {
				continue;
			}

			int y_ab = 0;		// observation
			Edge edge(std::min(i, neighbor), std::max(i, neighbor));
			if (edge.in(network.get_linked_edges())) {
				y_ab = 1;
			}

			std::vector<double> probs(K);
#if ! defined EFFICIENCY_FOLLOWS_CPP_WENZHE
			double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
#endif
			for (::size_t k = 0; k < K; k++) {
#if defined EFFICIENCY_FOLLOWS_CPP_WENZHE
				probs[k] = std::pow(beta[k], y_ab) * std::pow(1 - beta[k], 1 - y_ab) * pi[i][k] * pi[neighbor][k];
				probs[k] += std::pow(epsilon, y_ab) * std::pow(1 - epsilon, 1 - y_ab) * pi[i][k] * (1 - pi[neighbor][k]);
#else
				double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
				probs[k] = pi[i][k] * (pi[neighbor][k] * f + e);
#endif
			}

			double prob_sum = np::sum(probs);
			for (::size_t k = 0; k < K; k++) {
				grads[k] += (probs[k] / prob_sum) / phi[i][k] - 1.0 / phi_i_sum;
			}
		}

		std::vector<double> noise = kernelRandom->randn(K);	// random gaussian noise.
		if (false) {
			for (::size_t k = 0; k < K; ++k) {
				std::cerr << "randn " << std::fixed << std::setprecision(12) << noise[k] << std::endl;
			}
		}
#ifndef EFFICIENCY_FOLLOWS_CPP_WENZHE
		double Nn = (1.0 * N) / num_node_sample;
#endif
        // update phi for node i
        for (::size_t k = 0; k < K; k++) {
#ifdef EFFICIENCY_FOLLOWS_CPP_WENZHE
			// FIXME replace a**0.5 * b**0.5 with sqrt(a * b)
			phi[i][k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] + \
														  (N*1.0 / num_node_sample) * grads[k]) +
								 std::pow(eps_t, 0.5) * std::pow(phi[i][k], 0.5) * noise[k]);
#else
			phi[i][k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] + \
														  Nn * grads[k]) +
								 sqrt(eps_t * phi[i][k]) * noise[k]);
#endif
		}

		if (false) {
			std::cerr << std::fixed << std::setprecision(12) << "update_phi post Nn " << Nn << " phi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
			}
			std::cerr << std::endl;
			std::cerr << "pi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
			}
			std::cerr << std::endl;
			std::cerr << "grads ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << grads[k] << " ";
			}
			std::cerr << std::endl;
			std::cerr << "noise ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << noise[k] << " ";
			}
			std::cerr << std::endl;
		}

		// assign back to phi.
		//phi[i] = phi_star;
	}


#if 0
    def __sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
        /**
        we need to calculate z_ab. We can use deterministic way to calculate this
        for each k,  p[k] = p(z_ab=k|*) = \sum_{i}^{} p(z_ab=k, z_ba=i|*)
        then we simply sample z_ab based on the distribution p.
        this runs in O(K)
         */
        p = np.zeros(K)
        for i in range(0, K):
            tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
            tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
            p[i] = tmp
        // sample community based on probability distribution p.
        bounds = np.cumsum(p)
        location = random.random() * bounds[K-1]

        // get the index of bounds that containing location.
        for i in range(0, K):
                if location <= bounds[i]:
                    return i
        // failed, should not happen!
        return -1
#endif

	// TODO FIXME make VertexSet an out parameter
    NeighborSet sample_neighbor_nodes(::size_t sample_size, int nodeId) {
        /**
        Sample subset of neighborhood nodes.
         */
        int p = (int)sample_size;
        NeighborSet neighbor_nodes;
        const EdgeMap &held_out_set = network.get_held_out_set();
        const EdgeMap &test_set = network.get_test_set();

#if defined RANDOM_FOLLOWS_CPP_WENZHE || defined EFFICIENCY_FOLLOWS_PYTHON
        while (p > 0) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
			std::cerr << "FIXME: horribly inefficient xrange thingy" << std::endl;
			auto nodeList = Random::random->sample(np::xrange(0, N), sample_size * 2);
#else
			auto nodeList = Random::random->sampleRange(N, sample_size * 2);
#endif

            for (std::vector<int>::const_iterator neighborId = nodeList->begin();
				 	neighborId != nodeList->end();
					neighborId++) {
				if (p < 0) {
					if (p != 0) {
						// std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					}
					break;
				}
				if (*neighborId == nodeId) {
					continue;
				}
				// check condition, and insert into mini_batch_set if it is valid.
				Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
				if (edge.in(held_out_set) || edge.in(test_set) ||
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
						find(neighbor_nodes.begin(), neighbor_nodes.end(), neighborId) != neighbor_nodes.end()
#else
						neighbor_nodes.find(*neighborId) != neighbor_nodes.end()
#endif
						) {
					continue;
				} else {
					// add it into mini_batch_set
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
					neighbor_nodes.push_back(*neighborId);
#else
					neighbor_nodes.insert(*neighborId);
#endif
					p -= 1;
				}
			}

			delete nodeList;
		}
#else	// if defined RANDOM_FOLLOWS_CPP_WENZHE || defined EFFICIENCY_FOLLOWS_PYTHON
        for (int i = 0; i <= p; ++i) {
			int neighborId;
			Edge edge(0, 0);
			do {
				neighborId = kernelRandom->randint(0, N - 1);
				edge = Edge(std::min(nodeId, neighborId), std::max(nodeId, neighborId));
				// std::cerr << std::fixed << std::setprecision(12) << "node " << nodeId << " neighbor " << neighborId << " peer " << ! (edge.in(held_out_set) || edge.in(test_set)) << " randint " << neighborId << " seed " << kernelRandom->state() << std::endl;
			} while (neighborId == nodeId
					|| edge.in(held_out_set)
					|| edge.in(test_set)
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
					|| find(neighbor_nodes.begin(), neighbor_nodes.end(), neighborId) != neighbor_nodes.end()
#else
					|| neighbor_nodes.find(neighborId) != neighbor_nodes.end()
#endif
					);
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
			neighbor_nodes.push_back(neighborId);
#else
			neighbor_nodes.insert(neighborId);
#endif
		}
#endif
		if (false) {
			std::cerr << "Node " << nodeId << ": neighbors ";
			for (auto n : neighbor_nodes) {
				std::cerr << n << " ";
			}
			std::cerr << std::endl;
		}

		return neighbor_nodes;
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

#if 0
    def _save(self):
        f = open('ppx_mcmc.txt', 'wb')
        for i in range(0, len(self._avg_log)):
            f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) +"\n")
        f.close()
#endif


#if 0
	// in this case, we need to sample randoms from kernelRandom
    int sample_z_ab_from_edge(int y,
							  const std::vector<double> &pi_a,
							  const std::vector<double> &pi_b,
							  const std::vector<double> &beta,
							  double epsilon, ::size_t K) {
		std::vector<double> p(K);

#ifdef EFFICIENCY_FOLLOWS_PYTHON
        for (::size_t i = 0; i < K; i++) {
			// FIMXE lift common expressions
            double tmp = std::pow(beta[i], y) * std::pow(1-beta[i], 1-y) * pi_a[i] * pi_b[i];
            // tmp += std::pow(epsilon, y) * std::pow(1-epsilon, 1-y) * pi_a[i] * (1 - pi_b[i]);
			double fac = std::pow(epsilon, y) * std::pow(1.0 - epsilon, 1 - y);
            tmp += fac * pi_a[i] * (1 - pi_b[i]);
            p[i] = tmp;
		}
#else
		if (y == 1) {
			for (::size_t i = 0; i < K; i++) {
				// p[i] = beta[i] * pi_a[i] * pi_b[i] + epsilon * pi_a[i] * (1 - pi_b[i])
				//      = pi_a[i] * (beta[i] * pi_b[i] + epsilon * (1 - pi_b[i]))
				//      = pi_a[i] * (pi_b[i] * (beta[i] - epsilon) + epsilon)
				p[i] = pi_a[i] * (pi_b[i] * (beta[i] - epsilon) + epsilon);
			}
		} else {
			double one_eps = 1.0 - epsilon;
			for (::size_t i = 0; i < K; i++) {
				// p[i] = (1 - beta[i]) * pi_a[i] * pi_b[i] + (1 - epsilon) * pi_a[i] * (1 - pi_b[i])
				//      = pi_a[i] * ((1 - beta[i]) * pi_b[i] + (1 - epsilon) * (1 - pi_b[i]))
				//      = pi_a[i] * (pi_b[i] * (1 - beta[i] - (1 - epsilon)) + (1 - epsilon) * 1)
				//      = pi_a[i] * (pi_b[i] * (-beta[i] + epsilon) + 1 - epsilon)
				p[i] = pi_a[i] * (pi_b[i] * (epsilon - beta[i]) + one_eps);
			}
		}
#endif

        for (::size_t k = 1; k < K; k++) {
            p[k] += p[k-1];
		}

        double r = kernelRandom->random();
        double location = r * p[K-1];
#if 0
        // get the index of bounds that containing location.
        for (::size_t i = 0; i < K; i++) {
            if (location <= p[i]) {
                return i;
			}
		}

        // failed, should not happen!
        return -1;
#else
		return np::find_le(p, location);
#endif
	}
#endif


protected:
	// replicated in both mcmc_sampler_
	double	a;
	double	b;
	double	c;

	::size_t num_node_sample;
	::size_t interval;

	std::vector<std::vector<double> > theta;		// parameterization for \beta
	std::vector<std::vector<double> > phi;			// parameterization for \pi
	const Options &args;
	Random::Random *kernelRandom;
};

}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
