#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/sample_latent_vars.h"

#include "mcmc/learning/learner.h"
#include "mcmc/learning/mcmc_sampler.h"

namespace mcmc {
namespace learning {

// typedef std::unordered_map<Edge, int>	EdgeMapZ;
typedef std::map<Edge, int>	EdgeMapZ;

class MCMCSamplerStochastic : virtual public MCMCSampler {
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
    MCMCSamplerStochastic(const Options &args, const Network &network)
			: Learner(args, network), MCMCSampler(args, network, N / 5, 100.0, 0.01) {
	}

	virtual ~MCMCSamplerStochastic() {
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

		if (step_count % 1 == 0) {
			double ppx_score = cal_perplexity_held_out();
			std::cout << std::fixed << std::setprecision(15) << "perplexity for hold out set is: " << ppx_score << std::endl;
			ppxs_held_out.push_back(ppx_score);
#if 0
			if (ppx_score < 5.0) {
				stepsize_switch = true;
				//print "switching to smaller step size mode!"
			}
#endif
		}

        while (step_count < max_iteration && ! is_converged()) {
			auto l1 = std::chrono::system_clock::now();
            //print "step: " + str(self._step_count)
            /**
            pr = cProfile.Profile()
            pr.enable()
             */

            // (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			std::unordered_map<int, std::vector<int> > latent_vars;
			std::unordered_map<int, ::size_t> size;

            // iterate through each node in the mini batch.
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);

			sample_latent_vars_stub(nodes, size, latent_vars);

			update_pi_for_node_stub(nodes, size, latent_vars, scale);

            // sample (z_ab, z_ba) for each edge in the mini_batch.
            // z is map structure. i.e  z = {(1,10):3, (2,4):-1}
			EdgeMapZ z = sample_latent_vars2(mini_batch);
            update_beta(mini_batch, scale, z);


            if (step_count % 1 == 0) {
                double ppx_score = cal_perplexity_held_out();
				std::cout << std::fixed << std::setprecision(12) << "perplexity for hold out set is: " << ppx_score << std::endl;
                ppxs_held_out.push_back(ppx_score);
#if 0
                if (ppx_score < 5.0) {
                    stepsize_switch = true;
                    //print "switching to smaller step size mode!"
				}
#endif
			}

			delete edgeSample.first;

            step_count++;
			auto l2 = std::chrono::system_clock::now();
			std::cout << "LOOP  = " << (l2-l1).count() << std::endl;

            /**
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
             */
		}
	}


protected:

    virtual void sample_latent_vars_stub(const OrderedVertexSet& nodes,
    			std::unordered_map<int, ::size_t>& size,
    			std::unordered_map<int, std::vector<int> >& latent_vars) {
    	for (auto node = nodes.begin();
				node != nodes.end();
				node++) {
			// sample a mini-batch of neighbors
			OrderedVertexSet neighbor_nodes = sample_neighbor_nodes(num_node_sample, *node);
			size[*node] = neighbor_nodes.size();
			// sample latent variables z_ab for each pair of nodes
			std::vector<int> z = sample_latent_vars(*node, neighbor_nodes);
			// save for a while, in order to update together.
			latent_vars[*node] = z;
		}
    }

    virtual void update_pi_for_node_stub(const OrderedVertexSet& nodes,
			std::unordered_map<int, ::size_t>& size,
			std::unordered_map<int, std::vector<int> >& latent_vars,
			double scale) {
    	// update pi for each node
		for (auto node = nodes.begin();
				node != nodes.end();
				node++) {
			// FIXME: misnomer, should be: update_phi_for_node
			update_pi_for_node(*node, latent_vars[*node], size[*node], scale);
		}
    }

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


	// FIXME lots of code sharing w/ mcmc_sampler_batch
    void update_beta(const OrderedEdgeSet &mini_batch, double scale, const EdgeMapZ &z) {
        /**
        update beta for mini_batch.
         */
		double eps_t = a * std::pow(1.0 + step_count / b, -c);

		std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));	// gradients K*2 dimension
        // sums = np.sum(self.__theta,1)
		std::vector<double> sums(theta.size());
		std::transform(theta.begin(), theta.end(), sums.begin(), np::sum<double>);
		std::vector<std::vector<double> > noise = Random::random->randn(K, 2);	// random noise.

        for (EdgeMapZ::const_iterator edge = z.begin();
			 	edge != z.end();
				edge++) {
            int y_ab = 0;
            if (edge->first.in(network.get_linked_edges())) {
                y_ab = 1;
			}
            int k = edge->second;
            // if k==-1 means z_ab != z_ba => gradient is 0.
            if (k == -1) {
                continue;
			}

            grads[k][0] += std::abs(1-y_ab) / theta[k][0] - 1 / sums[k];
            grads[k][1] += std::abs(-y_ab) / theta[k][1] - 1 / sums[k];
		}

        //if (mini_batch.size() < 1) {
		//	scale = 1;
		//} else {
        //	scale = (N * (N-1)/2)/mini_batch.size();
		//}
        // update theta
		std::vector<std::vector<double> > theta_star(theta);
        for (::size_t k = 0; k < K; k++) {
            for (::size_t i = 0; i < 2; i++) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
				// FIXME rewrite a**0.5 * b**0.5 as sqrt(a * b)
                theta_star[k][i] = std::abs(theta[k][i] + eps_t * (eta[i] - theta[k][i] + \
																   scale * grads[k][i]) +
										   	std::pow(2.0 * eps_t, .5) * std::pow(theta[k][i], .5) * noise[k][i]);
#else
                theta_star[k][i] = std::abs(theta[k][i] + eps_t * (eta[i] - theta[k][i] + \
																   scale * grads[k][i]) +
										   	sqrt(2.0 * eps_t * theta[k][i]) * noise[k][i]);
#endif
			}
		}

		theta = theta_star;
		// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
		// self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
	}


	// FIXME: misnomer, should be: update_phi_for_node
	// FIXME: scale is unused
    void update_pi_for_node(int i, const std::vector<int> &z, int n, double scale) {
        /**
        update pi for current node i.
         */
        // update gamma, only update node in the grad
		double eps_t;
        // if (! stepsize_switch) {
        //     eps_t = std::pow(1024+step_count, -0.5);
		// } else {
            eps_t  = a * std::pow(1 + step_count / b, -c);
		// }

		// FIXME: no need to initialize phi_star
		// std::vector<double> phi_star(phi[i]);					// updated \phi
		std::vector<double> phi_star(phi[i].size());			// updated \phi
		double phi_i_sum = np::sum(phi[i]);
		std::vector<double> noise = Random::random->randn(K);	// random noise.

        // get the gradients
        // grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        std::vector<double> grads(K, -n * 1.0/phi_i_sum);
        for (::size_t k = 0; k < K; k++) {
            grads[k] += 1.0 / phi[i][k] * z[k];
		}

        // update the phi
        for (::size_t k = 0; k < K; k++) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
			// FIXME replace a**0.5 * b**0.5 with sqrt(a * b)
            phi_star[k] = std::abs(phi[i][k] + eps_t/2 * (alpha - phi[i][k] + \
														  (N/n) * grads[k]) +
								   std::pow(eps_t, .5) * std::pow(phi[i][k], .5) * noise[k]);
#else
            phi_star[k] = std::abs(phi[i][k] + eps_t/2 * (alpha - phi[i][k] + \
														  (N/n) * grads[k]) +
								   sqrt(eps_t * phi[i][k]) * noise[k]);
#endif
		}

        // self.__phi[i] = phi_star
		phi[i] = phi_star;
        //self.__phi[i] = phi_star * (1.0/(self._step_count+1)) + (1-1.0/(self._step_count+1))*self.__phi[i]

        // update pi
        // double sum_phi = np::sum(phi[i]);
        // self._pi[i] = [self.__phi[i,k]/sum_phi for k in range(0, self._K)]
		np::normalize(&pi[i], phi[i]);
	}


    virtual EdgeMapZ sample_latent_vars2(const OrderedEdgeSet &mini_batch) {
        /**
        sample latent variable (z_ab, z_ba) for each pair of nodes. But we only consider 11 different cases,
        since we only need indicator function in the gradient update. More details, please see the comments
        within the sample_z_for_each_edge function.
         */
		EdgeMapZ z;
		for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
            int y_ab = 0;
            if (edge->in(network.get_linked_edges())) {
                y_ab = 1;
			}

            z[*edge] = sample_z_for_each_edge(y_ab, pi[edge->first], pi[edge->second], \
											  beta, K);
		}

        return z;
	}


	// TODO FIXME shared code w/ mcmc_sampler_batch
    int sample_z_for_each_edge(int y, const std::vector<double> &pi_a, const std::vector<double> &pi_b, const std::vector<double> &beta, ::size_t K) const {
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
#ifdef EFFICIENCY_FOLLOWS_PYTHON
		for (::size_t k = 0; k < K; k++) {
            p[k] = std::pow(beta[k], y) * pow(1-beta[k], 1-y) * pi_a[k] * pi_b[k];
		}
#else
		if (y == 1) {
			for (::size_t k = 0; k < K; k++) {
				p[k] = beta[k] * pi_a[k] * pi_b[k];
			}
		} else {
			for (::size_t k = 0; k < K; k++) {
				p[k] = (1-beta[k]) * pi_a[k] * pi_b[k];
			}
		}
#endif
        // p[K] = 1 - np.sum(p[0:K])
        p[K] = 1.0 - std::accumulate(p.begin(), p.begin() + K, 0.0);

        // sample community based on probability distribution p.
		for (::size_t k = 1; k < K + 1; k++) {
			p[k] += p[k - 1];
		}
        // // bounds = np.cumsum(p)
		// // std::vector<double> bounds(K + 1);
		// // std::partial_sum(p.begin(), p.end(), bounds.begin());
		// FIXME: replace p[K] w/ p[K-1] here. Why? RFHH
        // double location = Random::random->random() * p[K];
        double r = Random::random->random();
        double location = r * p[K-1];

#if 0
        // get the index of bounds that containing location.
        for (::size_t i = 0; i < K; i++) {
			if (location <= p[i]) {
				return i;
			}
		}

        return -1;
#else
		return np::find_le(p, location, K);
#endif
	}


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
    OrderedVertexSet sample_neighbor_nodes(::size_t sample_size, int nodeId) {
        /**
        Sample subset of neighborhood nodes.
         */
        int p = (int)sample_size;
        OrderedVertexSet neighbor_nodes;
        const EdgeMap &held_out_set = network.get_held_out_set();
        const EdgeMap &test_set = network.get_test_set();

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
//						std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					}
					break;
				}
				if (*neighborId == nodeId) {
					continue;
				}
				// check condition, and insert into mini_batch_set if it is valid.
				Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
				if (edge.in(held_out_set) || edge.in(test_set) || neighbor_nodes.find(*neighborId) != neighbor_nodes.end()) {
					continue;
				} else {
					// add it into mini_batch_set
					neighbor_nodes.insert(*neighborId);
					p -= 1;
				}
			}

			delete nodeList;
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


#ifdef SPOT_NO_DIFFERENCE_WITH_COMMON
    int sample_z_ab_from_edge(int y,
							  const std::vector<double> &pi_a,
							  const std::vector<double> &pi_b,
							  const std::vector<double> &beta,
							  double epsilon, ::size_t K) const {
		std::vector<double> p(K);

#ifdef EFFICIENCY_FOLLOWS_PYTHON
        for (::size_t i = 0; i < K; i++) {
            double tmp = std::pow(beta[i], y) * std::pow(1-beta[i], 1-y) * pi_a[i] * pi_b[i];
            tmp += std::pow(epsilon, y) * std::pow(1-epsilon, 1-y) * pi_a[i] * (1 - pi_b[i]);
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

        double r = Random::random->random();
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
};

}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
