#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include <mpi.h>

#include <d-kv-store/DKVStore.h>

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"

#include "mcmc/learning/learner.h"
#include "mcmc/learning/mcmc_sampler_stochastic.h"

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


/**
 * The distributed version differs in these aspects from the parallel version:
 *  - the minibatch is distributed
 *  - pi/phi is distributed
 *
 * Algorithm:
 * LOOP:
 *   One host (the Master) creates the minibatch and distributes its nodes over
 *   the workers. Each worker samples a neighbor set for each of its minibatch nodes.
 *
 *   Then, each worker:
 *    - fetches pi/phi for its minibatch nodes
 *    - fetches pi/phi for the neighbors of its minibatch nodes
 *    - calulates an updated phi/pi for its minibatch nodes
 *    - stores the updated phi/pi
 *
 *   Then, all synchronise. The master calculates an updated value for beta. This could
 *   plausibly be done in a distributed way, but it is so quick that we guess there is
 *   no point to do that.
 *   If needs, the master calculates the perplexity. If termination is met,
 *   let the workers know. Else, the master broadcasts its updated value for beta.
 * END LOOP
 */
class MCMCSamplerStochasticDistributed : public MCMCSamplerStochastic {
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
    MCMCSamplerStochasticDistributed(const Options &args, const Network &graph)
			: MCMCSamplerStochastic(args, graph) {
		// FIXME FIXME FIXME
		// TODO TODO TODO
		// Make kernelRandom init depend on mpi_rank
		// FIXME FIXME FIXME
		t_outer = timer::Timer("  outer");
		t_perplexity = timer::Timer("  perplexity");
		t_mini_batch = timer::Timer("  sample_mini_batch");
		t_nodes_in_mini_batch = timer::Timer("  nodes_in_mini_batch");
		t_sample_neighbor_nodes = timer::Timer("  sample_neighbor_nodes");
		t_update_phi = timer::Timer("  update_phi");
		t_update_pi = timer::Timer("  update_pi");
		t_update_beta = timer::Timer("  update_beta");
		timer::Timer::setTabular(true);
	}

	virtual ~MCMCSamplerStochasticDistributed() {
		(void)MPI_Finalize();
		std::cerr << "FIXME: close off d-kv-store" << std::endl;
	}

	virtual void init() {
		int r;

		int provided;
		r = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
		mpi_error_test(r, "MPI_Init_thread() fails");
		if (provided < MPI_THREAD_MULTIPLE) {
			mpi_error_test(MPI_ERR_ARG, "No multithreaded support: provide " +
						   boost::lexical_cast<std::string>(provided));
		}

		r = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
		mpi_error_test(r, "MPI_Comm_set_errhandler fails");

		r = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		mpi_error_test(r, "MPI_Comm_size() fails");
		r = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		mpi_error_test(r, "MPI_Comm_rank() fails");

#ifdef RANDOM_FOLLOWS_CPP_WENZHE
#  error No support for Wenzhe Random compatibility
#endif
		kernelRandom = new Random::Random(mpi_rank * 42);

		if (mpi_rank == mpi_master) {
			init_beta();
		}
		// TODO Master broadcasts beta

		init_pi();
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

    virtual void run() {
        /** run mini-batch based MCMC sampler, based on the sungjin's note */

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
			if (mpi_rank == mpi_master) {
				if (step_count % interval == 0) {
					// TODO Only at the Master
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
					// TODO Master broadcasts result to the workers
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
			}

			int r = MPI_Bcast(beta.data(), beta.size() * sizeof beta[0], MPI_DOUBLE, mpi_master, MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Bcast of beta fails");

			std::vector<int32_t> nodes_vector;
			EdgeSample edgeSample = deploy_mini_batch(&nodes_vector);

#ifndef EFFICIENCY_FOLLOWS_CPP_WENZHE
			double eps_t  = a * std::pow(1 + step_count / b, -c);	// step size
			// double eps_t = std::pow(1024+step_count, -0.5);
#endif

			// ************ do in parallel at each host
			// std::cerr << "Sample neighbor nodes" << std::endl;
			// FIXME: nodes_in_batch should generate a vector, not an OrderedVertexSet
#pragma omp parallel for
			for (::size_t n = 0; n < nodes_vector.size(); ++n) {
				int node = nodes_vector[n];
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

			// TODO calculate and store updated values for pi/phi
			// ************ do in parallel at each host
			t_update_pi.start();
#if defined EFFICIENCY_FOLLOWS_CPP_WENZHE
			// std::cerr << __func__ << ":" << __LINE__ << ":  FIXME" << std::endl;
			np::row_normalize(&pi, phi);	// update pi from phi.
#else
			// No need to update pi where phi is unchanged
			for (auto i: nodes_vector) {
				np::normalize(&pi[i], phi[i]);
			}
#endif
			t_update_pi.stop();
			std::cerr << "FIXME FIXME: write back phi/pi after update" << std::endl;

			// TODO
			// all synchronize with barrier

			// TODO
			// Only at the Master
			if (mpi_rank == mpi_master) {
				t_update_beta.start();
				update_beta(*edgeSample.first, edgeSample.second);
				t_update_beta.stop();

				// TODO FIXME allocate this outside the loop
				delete edgeSample.first;
			}

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
	void init_beta() {
		// TODO only at the Master
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
	}


	void init_pi() {
		for (int32_t i = mpi_rank; i < static_cast<int32_t>(N); i += mpi_size) {
			std::vector<double> phi_pi = kernelRandom->gamma(1, 1, 1, K)[0];
			if (true) {
				if (i < 10) {
					std::cerr << "phi[" << i << "]: ";
					for (::size_t k = 0; k < std::min(K, 10UL); k++) {
						std::cerr << std::fixed << std::setprecision(12) << phi_pi[k] << " ";
					}
					std::cerr << std::endl;
				}
			}
#ifndef NDEBUG
			for (auto ph : phi_pi) {
				assert(ph >= 0.0);
			}
#endif

			double phi_sum = std::accumulate(phi_pi.begin(), phi_pi.end(), 0.0);
			std::transform(phi_pi.begin(), phi_pi.end(), phi_pi.begin(),
						   [phi_sum](double p) {
							   return p / phi_sum;
						   });
			if (true) {
				if (i < 10) {
					std::cerr << "pi[" << i << "]: ";
					for (::size_t k = 0; k < std::min(K, 10UL); k++) {
						std::cerr << std::fixed << std::setprecision(12) << phi_pi[k] << " ";
					}
					std::cerr << std::endl;
				}
			}

			phi_pi.resize(K + 1);
			phi_pi[K] = phi_sum;
			std::vector<int32_t> node(1, i);
			std::vector<const double *> pi(1, phi_pi.data());
			d_kv_store->WriteKVRecords(node, pi);
		}
	};


	EdgeSample deploy_mini_batch(std::vector<int32_t> *nodes_vector) {
		std::vector<int32_t> minibatch_chunk(mpi_size);	// used only at Master	FIXME: lift to class
		std::vector<int32_t> scatter_minibatch;			// used only at Master	FIXME: lift to class
		std::vector<int32_t> scatter_displs(mpi_size);	// used only at Master	FIXME: lift to class
		int		r;
		EdgeSample edgeSample;

		if (mpi_rank == mpi_master) {
			// TODO Only at the Master
			// (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			// std::cerr << "Invoke sample_mini_batch" << std::endl;
			// TODO FIXME allocate edgeSample.first statically, and pass as parameter
			t_mini_batch.start();
			edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
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

			//std::unordered_map<int, std::vector<int> > latent_vars;
			//std::unordered_map<int, ::size_t> size;

			// iterate through each node in the mini batch.
			t_nodes_in_mini_batch.start();
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();
// std::cerr << "mini_batch size " << mini_batch.size() << " num_node_sample " << num_node_sample << std::endl;

			std::vector<std::vector<int>> subminibatch;

			::size_t upper_bound = (nodes.size() + 1 - mpi_size) / mpi_size;
			std::unordered_set<int> unassigned;
			for (auto n: nodes) {
				::size_t owner = node_owner(n);
				if (subminibatch[owner].size() == upper_bound) {
					unassigned.insert(n);
				} else {
					subminibatch[owner].push_back(n);
				}
			}

			::size_t i = 0;
			for (auto n: unassigned) {
				while (subminibatch[i].size() == upper_bound) {
					i++;
					assert(i < mpi_size);
				}
				subminibatch[i].push_back(n);
			}

			scatter_minibatch.clear();
			int32_t running_sum = 0;
			for (int i = 0; i < mpi_size; i++) {
				minibatch_chunk[i] = subminibatch[i].size();
				scatter_displs[i] = running_sum;
				running_sum += subminibatch[i].size();
				scatter_minibatch.insert(scatter_minibatch.end(),
										 subminibatch[i].begin(),
										 subminibatch[i].end());
			}
		}

		int32_t my_minibatch_size;
		r = MPI_Scatter(minibatch_chunk.data(), 1, MPI_INT,
						&my_minibatch_size, 1, MPI_INT,
						mpi_master, MPI_COMM_WORLD);
		mpi_error_test(r, "MPI_Scatter of minibatch chunks fails");
		nodes_vector->resize(my_minibatch_size);

		if (mpi_rank == mpi_master) {
			// TODO Master scatters the minibatch nodes over the workers,
			// preferably with consideration for both load balance and locality
			r = MPI_Scatterv(scatter_minibatch.data(),
							 minibatch_chunk.data(),
							 scatter_displs.data(),
							 MPI_INT,
							 nodes_vector->data(),
							 my_minibatch_size,
							 MPI_INT,
							 mpi_master,
							 MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Scatterv of minibatch fails");

		} else {
			r = MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
							 nodes_vector->data(), my_minibatch_size, MPI_INT,
							 mpi_master, MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Scatterv of minibatch fails");
		}

		return edgeSample;
	}

	int node_owner(int node) const {
		return node % mpi_size;
	}


	static void mpi_error_test(int r, const std::string &message) {
		if (r != MPI_SUCCESS) {
			throw MCMCException("MPI error " + r + message);
		}
	}


protected:
	const int mpi_master = 0;
	int		mpi_size;
	int		mpi_rank;

	DKV::DKVStoreInterface *d_kv_store;

	timer::Timer t_outer;
	timer::Timer t_perplexity;
	timer::Timer t_mini_batch;
	timer::Timer t_nodes_in_mini_batch;
	timer::Timer t_sample_neighbor_nodes;
	timer::Timer t_update_phi;
	timer::Timer t_update_pi;
	timer::Timer t_update_beta;
};

}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
