#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include <mpi.h>

#include <d-kv-store/DKVStore.h>
#include <d-kv-store/file/DKVStoreFile.h>
#ifdef ENABLE_RAMCLOUD
#include <d-kv-store/ramcloud/DKVStoreRamCloud.h>
#endif
#ifdef ENABLE_RDMA
#include <d-kv-store/ramcloud/DKVStoreRamCloud.h>
#endif

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
 *  - pi/phi is distributed. It is not necessary to store both phi and pi since
 *    the two are equivalent. We store pi + sum(phi), and we can restore phi
 *    as phi[i] = pi[i] * phi_sum.
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
			: MCMCSamplerStochastic(args, graph), args(args) {
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
		throw MCMCException("No support for Wenzhe Random compatibility");
#endif
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		throw MCMCException("No support for Scalable-Graph Random compatibility");
#endif
#ifdef EFFICIENCY_FOLLOWS_CPP_WENZHE
		throw MCMCException("No support for Wenzhe Efficiency compatibility");
#endif
#ifdef EFFICIENCY_FOLLOWS_PYTHON
		throw MCMCException("No support for Python Efficiency compatibility");
#endif

		t_outer = timer::Timer("  outer");
		t_perplexity = timer::Timer("  perplexity");
		t_mini_batch = timer::Timer("  sample_mini_batch");
		t_nodes_in_mini_batch = timer::Timer("  nodes_in_mini_batch");
		t_sample_neighbor_nodes = timer::Timer("  sample_neighbor_nodes");
		t_update_phi = timer::Timer("  update_phi");
		t_update_pi = timer::Timer("  update_pi");
		t_update_beta = timer::Timer("  update_beta");
		t_pi_read_node = timer::Timer("  load minibatch pi");
		t_pi_read_neighbor = timer::Timer("  load neighbor pi");
		timer::Timer::setTabular(true);
	}


	virtual ~MCMCSamplerStochasticDistributed() {
		(void)MPI_Finalize();
		std::cerr << "FIXME: close off d-kv-store" << std::endl;

		delete d_kv_store;
	}


	virtual void init() {
		int r;

		int provided;
		r = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
		mpi_error_test(r, "MPI_Init_thread() fails");
		if (provided < MPI_THREAD_MULTIPLE) {
#ifdef NOT_NOW_NEED_TO_RUN_VALGRIND
			mpi_error_test(MPI_ERR_ARG, "No multithreaded support: provide " +
						   boost::lexical_cast<std::string>(provided));
#else
			std::cerr << "No multithread MPI support. Works only for sequential runs" << std::endl;
#endif
		}

		r = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
		mpi_error_test(r, "MPI_Comm_set_errhandler fails");

		r = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		mpi_error_test(r, "MPI_Comm_size() fails");
		r = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		mpi_error_test(r, "MPI_Comm_rank() fails");

		// Make kernelRandom init depend on mpi_rank
		kernelRandom = new Random::Random((mpi_rank + 1) * 42);

		// d_kv_store = new DKV::DKVRamCloud::DKVStoreRamCloud();
		d_kv_store = new DKV::DKVFile::DKVStoreFile();

		max_minibatch_nodes = network.minibatch_nodes_for_strategy(mini_batch_size, strategy);
		max_minibatch_neighbors = max_minibatch_nodes * real_num_node_sample();
		std::cerr << "minibatch size param " << mini_batch_size << " max " << max_minibatch_nodes << " #neighbors(total) " << max_minibatch_neighbors << std::endl;

		d_kv_store->Init(K + 1,
						 N,
						 (max_minibatch_nodes + max_minibatch_neighbors + mpi_size - 1) / mpi_size,
						 args.getRemains());

		if (mpi_rank == mpi_master) {
			init_beta();
		}

		init_pi();
#ifdef DOESNT_WORK_FOR_DISTRIBUTED
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
#endif

        std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
		std::cerr << "Done constructor" << std::endl;
	}

    virtual void run() {
        /** run mini-batch based MCMC sampler, based on the sungjin's note */

        using namespace std::chrono;

		std::vector<std::vector<double>> phi_node(max_minibatch_nodes, std::vector<double>(K + 1));
		std::vector<double *> pi_node;
		std::vector<std::vector<double *>> pi_neighbor(max_minibatch_nodes,
													   std::vector<double *>(real_num_node_sample(),
																			 new double[K + 1]));

		t_start = clock();
        while (step_count < max_iteration && ! is_converged()) {

			t_outer.start();
			auto l1 = std::chrono::system_clock::now();
			//if (step_count > 200000){
				//interval = 2;
			//}
			if (mpi_rank == mpi_master) {
				check_perplexity();
			}

			int r = MPI_Bcast(beta.data(), beta.size() * sizeof beta[0], MPI_DOUBLE, mpi_master, MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Bcast of beta fails");

			std::vector<int32_t> nodes_vector;
			EdgeSample edgeSample = deploy_mini_batch(&nodes_vector);

			double eps_t  = a * std::pow(1 + step_count / b, -c);	// step size
			// double eps_t = std::pow(1024+step_count, -0.5);

			// ************ load our pi from D-KV store **************
            t_pi_read_node.start();
			pi_node.resize(nodes_vector.size());
            d_kv_store->ReadKVRecords(pi_node, nodes_vector, DKV::RW_MODE::READ_ONLY);
            t_pi_read_node.stop();
			// ************ do in parallel at each host
			// std::cerr << "Sample neighbor nodes" << std::endl;
			// FIXME: nodes_in_batch should generate a vector, not an OrderedVertexSet
#pragma omp parallel for
			for (::size_t i = 0; i < nodes_vector.size(); ++i) {
				int node = nodes_vector[i];
				t_sample_neighbor_nodes.start();
				// sample a mini-batch of neighbors
				NeighborSet neighbors = sample_neighbor_nodes(num_node_sample, node);
				t_sample_neighbor_nodes.stop();

                // ************ load neighor pi from D-KV store **********
                t_pi_read_neighbor.start();
#ifdef NEIGHBOR_SET_IS_VECTOR
                d_kv_store->ReadKVRecords(pi_neighbor[i], neighbors, DKV::RW_MODE::READ_ONLY);
#else
				std::vector<int32_t> neighbor_vector(neighbors.begin(), neighbors.end());
                d_kv_store->ReadKVRecords(pi_neighbor[i], neighbor_vector, DKV::RW_MODE::READ_ONLY);
#endif
                t_pi_read_neighbor.stop();

                // std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
				t_update_phi.start();
				update_phi(&phi_node[i], node, pi_node[i], neighbors, pi_neighbor[i], eps_t);
				t_update_phi.stop();
			}

			// all synchronize with barrier: ensure we read pi/phi_sum from current iteration
			MPI_Barrier(MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Barrier fails");

			// TODO calculate and store updated values for pi/phi_sum
#pragma omp parallel for
			for (::size_t i = 0; i < pi_node.size(); i++) {
				pi_from_phi(pi_node[i], phi_node[i]);
			}
			// std::cerr << "write back phi/pi after update" << std::endl;
			d_kv_store->WriteKVRecords(nodes_vector, constify(pi_node));
			d_kv_store->PurgeKVRecords();

			// all synchronize with barrier
			MPI_Barrier(MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Barrier fails");

			if (mpi_rank == mpi_master) {
				// TODO load pi/phi values for the minibatch nodes
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

		if (mpi_rank == mpi_master) {
			check_perplexity();
		}

		for (auto &p : pi_neighbor) {
			for (auto &n : p) {
				delete[] n;
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
		std::cout << t_pi_read_node << std::endl;
		std::cout << t_pi_read_neighbor << std::endl;
	}


protected:
	template <typename T>
	std::vector<const T*>& constify(std::vector<T*>& v) {
		// Compiler doesn't know how to automatically convert
		// std::vector<T*> to std::vector<T const*> because the way
		// the template system works means that in theory the two may
		// be specialised differently.  This is an explicit conversion.
		return reinterpret_cast<std::vector<const T*>&>(v);
	}


	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}


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


	// Calculate pi[0..K> ++ phi_sum from phi[0..K>
	void pi_from_phi(double *pi, const std::vector<double> &phi) {
		double phi_sum = std::accumulate(phi.begin(), phi.begin() + K, 0.0);
		for (::size_t k = 0; k < K; ++k) {
			pi[k] = phi[k] / phi_sum;
		}

		pi[K] = phi_sum;
	}


	void init_pi() {
		double pi[K + 1];
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

			pi_from_phi(pi, phi_pi);
			if (true) {
				if (i < 10) {
					std::cerr << "pi[" << i << "]: ";
					for (::size_t k = 0; k < std::min(K, 10UL); k++) {
						std::cerr << std::fixed << std::setprecision(12) << pi[k] << " ";
					}
					std::cerr << std::endl;
				}
			}

			std::vector<int32_t> node(1, i);
			std::vector<const double *> pi_wrapper(1, pi);
			d_kv_store->WriteKVRecords(node, pi_wrapper);
		}
	}


	void check_perplexity() {
		if (step_count % interval == 0) {
			t_perplexity.start();
			// TODO load pi for the held-out set to calculate perplexity
			double ppx_score = cal_perplexity_held_out();
			t_perplexity.stop();
			std::cout << std::fixed << std::setprecision(12) << "step count: " << step_count << " perplexity for hold out set: " << ppx_score << std::endl;
			ppxs_held_out.push_back(ppx_score);

			clock_t t2 = clock();
			double diff = (double)t2 - (double)t_start;
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
	}


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
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			if (true) {
				std::cerr << "Minibatch[" << mini_batch.size() << "]: ";
				for (auto e : mini_batch) {
					std::cerr << e << " ";
				}
				std::cerr << std::endl;
			}
			// std::cerr << "Done sample_mini_batch" << std::endl;

			//std::unordered_map<int, std::vector<int> > latent_vars;
			//std::unordered_map<int, ::size_t> size;

			// iterate through each node in the mini batch.
			t_nodes_in_mini_batch.start();
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();
// std::cerr << "mini_batch size " << mini_batch.size() << " num_node_sample " << num_node_sample << std::endl;

			std::vector<std::vector<int>> subminibatch(mpi_size);

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
			std::cerr << "#nodes " << nodes.size() << " #unassigned " << unassigned.size() << std::endl;

			::size_t i = 0;
			for (auto n: unassigned) {
				while (subminibatch[i].size() == upper_bound) {
					i++;
					assert(i < static_cast<::size_t>(mpi_size));
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


    void update_phi(std::vector<double> *phi_node,	// out parameter
					int i, const double *pi_node,
					const NeighborSet &neighbors, const std::vector<double *> &pi,
                    double eps_t) {
		if (false) {
			std::cerr << "update_phi pre ";
			std::cerr << "phi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << (pi_node[k] * pi_node[K]) << " ";
			}
			std::cerr << std::endl;
			std::cerr << "pi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << pi_node[k] << " ";
			}
			std::cerr << std::endl;
			::size_t ix = 0;
			for (auto n: neighbors) {
				std::cerr << "pi[" << n << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << pi[ix][k] << " ";
				}
				std::cerr << std::endl;
				ix++;
			}
		}

		double phi_i_sum = pi_node[K];
        std::vector<double> grads(K, 0.0);	// gradient for K classes
		// std::vector<double> phi_star(K);					// temp vars

		::size_t ix = 0;
		for (auto neighbor: neighbors) {
			if (i != neighbor) {
				int y_ab = 0;		// observation
				Edge edge(std::min(i, neighbor), std::max(i, neighbor));
				if (edge.in(network.get_linked_edges())) {
					y_ab = 1;
				}

				std::vector<double> probs(K);
				double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
				for (::size_t k = 0; k < K; k++) {
					double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
					probs[k] = pi_node[k] * (pi[ix][k] * f + e);
				}

				double prob_sum = np::sum(probs);
				for (::size_t k = 0; k < K; k++) {
					// grads[k] += (probs[k] / prob_sum) / phi[i][k] - 1.0 / phi_i_sum;
					grads[k] += ((probs[k] / prob_sum) / pi_node[k] - 1.0) / phi_i_sum;
				}
			}
			ix++;
		}

		std::vector<double> noise = kernelRandom->randn(K);	// random gaussian noise.
		if (false) {
			for (::size_t k = 0; k < K; ++k) {
				std::cerr << "randn " << std::fixed << std::setprecision(12) << noise[k] << std::endl;
			}
		}
		double Nn = (1.0 * N) / num_node_sample;
        // update phi for node i
        for (::size_t k = 0; k < K; k++) {
			double phi_node_k = pi_node[k] * phi_i_sum;
			(*phi_node)[k] = std::abs(phi_node_k + eps_t / 2 * (alpha - phi_node_k + \
															 Nn * grads[k]) +
								   sqrt(eps_t * phi_node_k) * noise[k]);
		}

		if (false) {
			std::cerr << std::fixed << std::setprecision(12) << "update_phi post Nn " << Nn << " phi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << (*phi_node)[k] << " ";
			}
			std::cerr << std::endl;
			std::cerr << "pi[" << i << "] ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << std::fixed << std::setprecision(12) << pi_node[k] << " ";
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


    void update_beta(const OrderedEdgeSet &mini_batch, double scale) {

		std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));	// gradients K*2 dimension
		std::vector<double> probs(K);
        // sums = np.sum(self.__theta,1)
		std::vector<double> theta_sum(theta.size());
		std::transform(theta.begin(), theta.end(), theta_sum.begin(), np::sum<double>);

		std::unordered_map<int, int> node_rank;
		std::vector<int> nodes;
		for (auto e : mini_batch) {
			int i = e.first;
			int j = e.second;
			if (node_rank.find(i) == node_rank.end()) {
				::size_t next = node_rank.size();
				node_rank[i] = next;
				nodes.push_back(i);
			}
			if (node_rank.find(j) == node_rank.end()) {
				::size_t next = node_rank.size();
				node_rank[j] = next;
				nodes.push_back(j);
			}
			assert(node_rank.size() == nodes.size());
		}
		std::vector<double *> pi(node_rank.size());
		d_kv_store->ReadKVRecords(pi, nodes, DKV::RW_MODE::READ_ONLY);

		// update gamma, only update node in the grad
		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		//double eps_t = std::pow(1024+step_count, -0.5);
		for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
            int y = 0;
            if (edge->in(network.get_linked_edges())) {
                y = 1;
			}
			int i = node_rank[edge->first];
			int j = node_rank[edge->second];

			double pi_sum = 0.0;
			for (::size_t k = 0; k < K; k++) {
				// Note: this is the KV-store cached pi, not the Learner item
				pi_sum += pi[i][k] * pi[j][k];
				double f = pi[i][k] * pi[j][k];
				if (y == 1) {
					probs[k] = beta[k] * f;
				} else {
					probs[k] = (1.0 - beta[k]) * f;
				}
			}

			double prob_0 = ((y == 1) ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
			double prob_sum = np::sum(probs) + prob_0;
			for (::size_t k = 0; k < K; k++) {
				double f = probs[k] / prob_sum;
				double one_over_theta_sum = 1.0 / theta_sum[k];
				grads[k][0] += f * ((1 - y) / theta[k][0] - one_over_theta_sum);
				grads[k][1] += f * (y / theta[k][1] - one_over_theta_sum);
			}
		}

        // update theta
		std::vector<std::vector<double> > noise = kernelRandom->randn(K, 2);	// random noise.
		// std::vector<std::vector<double> > theta_star(theta);
        for (::size_t k = 0; k < K; k++) {
            for (::size_t i = 0; i < 2; i++) {
				double f = std::sqrt(eps_t * theta[k][i]);
				theta[k][i] = std::abs(theta[k][i] +
									   eps_t / 2.0 * (eta[i] - theta[k][i] +
													  scale * grads[k][i]) +
									   f * noise[k][i]);
			}
		}

		// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
		// self._beta = temp[:,1]
		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));

		d_kv_store->PurgeKVRecords();

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
	double cal_perplexity_held_out() {
		const EdgeMap &data = network.get_held_out_set();

		double link_likelihood = 0.0;
		double non_link_likelihood = 0.0;
		::size_t link_count = 0;
		::size_t non_link_count = 0;

		// FIXME code duplication w/ update_beta
		// Is there also code duplication w/ update_phi?
		std::unordered_map<int, int> node_rank;
		std::vector<int> nodes;
		for (auto edge : data) {
			const Edge &e = edge.first;
			int i = e.first;
			int j = e.second;
			if (node_rank.find(i) == node_rank.end()) {
				::size_t next = node_rank.size();
				node_rank[i] = next;
				nodes.push_back(i);
			}
			if (node_rank.find(j) == node_rank.end()) {
				::size_t next = node_rank.size();
				node_rank[j] = next;
				nodes.push_back(j);
			}
			assert(node_rank.size() == nodes.size());
		}
		std::vector<double *> pi(node_rank.size());
		d_kv_store->ReadKVRecords(pi, nodes, DKV::RW_MODE::READ_ONLY);

		::size_t i = 0;
		for (auto edge : data) {
			const Edge &e = edge.first;
			int a = node_rank[e.first];
			int b = node_rank[e.second];
			double edge_likelihood = cal_edge_likelihood(pi[a], pi[b],
														 edge.second, beta);
			if (std::isnan(edge_likelihood)) {
				std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
			}

			//cout<<"AVERAGE COUNT: " <<average_count;
			ppx_for_heldout[i] = (ppx_for_heldout[i] * (average_count-1) + edge_likelihood)/(average_count);
			// std::cerr << std::fixed << std::setprecision(12) << e << " in? " << (e.in(network.get_linked_edges()) ? "True" : "False") << " -> " << edge_likelihood << " av. " << average_count << " ppx[" << i << "] " << ppx_for_heldout[i] << std::endl;
			// assert(edge->second == e.in(network.get_linked_edges()));
			if (edge.second) {
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

		d_kv_store->PurgeKVRecords();

		// std::cerr << std::setprecision(12) << "ratio " << link_ratio << " count: link " << link_count << " " << link_likelihood << " non-link " << non_link_count << " " << non_link_likelihood << std::endl;

		// weight each part proportionally.
		/*
		avg_likelihood = self._link_ratio*(link_likelihood/link_count) + \
		         (1-self._link_ratio)*(non_link_likelihood/non_link_count)
		*/

		// direct calculation.
		double avg_likelihood = 0.0;
		if (link_count + non_link_count != 0){
			avg_likelihood = (link_likelihood + non_link_likelihood) / (link_count + non_link_count);
		}
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


	int node_owner(int node) const {
		return node % mpi_size;
	}


	static void mpi_error_test(int r, const std::string &message) {
		if (r != MPI_SUCCESS) {
			throw MCMCException("MPI error " + r + message);
		}
	}


protected:
	::size_t	max_minibatch_nodes;
	::size_t	max_minibatch_neighbors;

	const int mpi_master = 0;
	int		mpi_size;
	int		mpi_rank;

	const Options &args;

	DKV::DKVStoreInterface *d_kv_store;

	timer::Timer t_outer;
	timer::Timer t_perplexity;
	timer::Timer t_mini_batch;
	timer::Timer t_nodes_in_mini_batch;
	timer::Timer t_sample_neighbor_nodes;
	timer::Timer t_update_phi;
	timer::Timer t_update_pi;
	timer::Timer t_update_beta;
	timer::Timer t_pi_read_node;
	timer::Timer t_pi_read_neighbor;

	clock_t	t_start;
	std::vector<double> timings;
};

}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
