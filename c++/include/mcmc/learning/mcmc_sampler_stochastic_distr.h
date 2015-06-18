#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#ifdef ENABLE_OPENMP
#  include <omp.h>
#else

static int omp_get_max_threads() {
	return 1;
}

static int omp_get_thread_num() {
	return 0;
}

static int omp_get_num_threads() {
	return 1;
}

#endif

#ifdef ENABLE_DISTRIBUTED
#  include <mpi.h>
#else

#define MPI_SUCCESS		0

typedef int MPI_Comm;
#define MPI_COMM_WORLD	0

enum MPI_ERRORS {
   	MPI_ERRORS_RETURN,
   	MPI_ERRORS_ARE_FATAL,
};

enum MPI_Datatype {
	MPI_INT                    = 0x4c000405,
	MPI_LONG                   = 0x4c000407,
	MPI_DOUBLE                 = 0x4c00080b,
	MPI_BYTE                   = 0x4c00010d,
	MPI_CUSTOM                 = 0x4c001000,
};

#define MPI_THREAD_SINGLE		0
#define MPI_THREAD_FUNNELED		1
#define MPI_THREAD_SERIALIZED	2
#define MPI_THREAD_MULTIPLE		3

int MPI_Init(int *argc, char ***argv) {
	return MPI_SUCCESS;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
	*provided = required;

	return MPI_SUCCESS;
}

struct MPI_TYPE {
	MPI_Datatype	datatype;
	::size_t		size;
};

static std::vector<MPI_TYPE> mpi_custom_type;
static MPI_Datatype custom_types = static_cast<MPI_Datatype>(MPI_CUSTOM + 1);

int MPI_Finalize() {
	return MPI_SUCCESS;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
			  int root, MPI_Comm comm) {
	return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm) {
	return MPI_SUCCESS;
}

int MPI_Comm_set_errhandler(MPI_Comm comm, int mode) {
	return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int *mpi_size_) {
	*mpi_size_ = 1;
	return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int *mpi_rank_) {
	*mpi_rank_ = 0;
	return MPI_SUCCESS;
}

::size_t mpi_datatype_size(MPI_Datatype type) {
	switch (type) {
	case MPI_INT:
		return sizeof(int32_t);
	case MPI_LONG:
		return sizeof(int64_t);
	case MPI_DOUBLE:
		return sizeof(double);
	case MPI_BYTE:
		return 1;
	default:
		if (type > MPI_CUSTOM && type < custom_types) {
			return mpi_custom_type[type - MPI_CUSTOM - 1].size;
		}
		std::cerr << "OOOPPPPPPSSS: Unknown datatype: " << (int)type << std::endl;
		return 0;
	}
}

int MPI_Type_contiguous(int n, MPI_Datatype base, MPI_Datatype *cont) {
	mpi_custom_type.resize(mpi_custom_type.size() + 1);
	mpi_custom_type[mpi_custom_type.size() - 1].datatype = custom_types;
	*cont = custom_types;
	custom_types = static_cast<MPI_Datatype>(custom_types + 1);
	mpi_custom_type[mpi_custom_type.size() - 1].size = n * mpi_datatype_size(base);
	return MPI_SUCCESS;
}

int MPI_Type_commit(MPI_Datatype *type) {
	return MPI_SUCCESS;
}

int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
				void *recvbuf, int recvcount, MPI_Datatype recvtype,
				int root, MPI_Comm comm) {
	memcpy(recvbuf, sendbuf, sendcount * mpi_datatype_size(sendtype));
	return MPI_SUCCESS;
}

int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
				 void *recvbuf, int recvcount, MPI_Datatype recvtype,
				 int root, MPI_Comm comm) {
	return MPI_Scatter((char *)sendbuf + displs[0] * mpi_datatype_size(sendtype),
					   sendcounts[0], sendtype,
					   recvbuf, recvcount, recvtype,
					   root, comm);
}

#endif

#include <d-kv-store/DKVStore.h>
#include <d-kv-store/file/DKVStoreFile.h>
#ifdef ENABLE_RAMCLOUD
#include <d-kv-store/ramcloud/DKVStoreRamCloud.h>
#endif
#ifdef ENABLE_RDMA
#include <d-kv-store/rdma/DKVStoreRDMA.h>
#endif

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"
#include <mr/timer.h>

#include "mcmc/learning/learner.h"
#include "mcmc/learning/mcmc_sampler_stochastic.h"

namespace DKV { namespace TYPE {
	enum DKV_TYPE {
		FILE,
#ifdef ENABLE_RAMCLOUD
		RAMCLOUD,
#endif
#ifdef ENABLE_RDMA
		RDMA,
#endif
	};
} }

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

using ::mr::timer::Timer;

struct WorkItem {
	WorkItem() {
	}

	WorkItem(int32_t m, int32_t n) : minibatch_node(m), neighbor_node(n) {
	}

	int32_t	minibatch_node;
	int32_t	neighbor_node;

	std::ostream &put(std::ostream &s) const {
		s << "<" << minibatch_node << "," << neighbor_node << ")";
		return s;
	}
};


inline std::ostream &operator << (std::ostream &s, const WorkItem &w) {
	return w.put(s);
}


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
 *   the workers. Each worker samples a neighbor_ set for each of its minibatch nodes.
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
			: MCMCSamplerStochastic(args, graph), mpi_master_(0), args(args) {
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

		std::cerr << "***************** FIXME: make an array of kernelRandoms, one for each of the OpenMP threads" << std::endl;

		t_populate_pi           = Timer("  populate pi");
		t_outer                 = Timer("  iteration");
		t_deploy_minibatch      = Timer("    deploy minibatch");
		t_mini_batch            = Timer("      sample_mini_batch");
		t_nodes_in_mini_batch   = Timer("      nodes_in_mini_batch");
		t_sample_neighbor_nodes = Timer("      sample_neighbor_nodes");
		t_load_pi_minibatch     = Timer("      load minibatch pi");
		t_load_pi_neighbor      = Timer("      load neighbor_ pi");
		t_grads_calc            = Timer("      calc grad chunks");
		t_grads_sum_local       = Timer("      locally sum grad chunks");
		t_grads_reduce          = Timer("      reduce(+) grad sums");
		t_update_phi            = Timer("      update_phi");
		t_barrier_phi           = Timer("    barrier to update phi");
		t_update_pi             = Timer("    update_pi");
		t_store_pi_minibatch    = Timer("      store minibatch pi");
		t_purge_pi_perp         = Timer("      purge perplexity pi");
		t_barrier_pi            = Timer("    barrier to update pi");
		t_update_beta           = Timer("    update_beta");
		t_load_pi_beta          = Timer("      load pi update_beta");
		t_broadcast_beta        = Timer("      broadcast beta");
		t_perplexity            = Timer("  perplexity");
		t_load_pi_perp          = Timer("      load perplexity pi");
		t_rank_pi_perp          = Timer("      rank pi perp");
		t_cal_edge_likelihood   = Timer("      calc edge likelihood");
		t_perp_log              = Timer("      calc log");
		Timer::setTabular(true);
	}


	virtual ~MCMCSamplerStochasticDistributed() {
		// std::cerr << "FIXME: close off d-kv-store" << std::endl;
		delete d_kv_store;

		for (auto r : threadRandom_) {
			delete r;
		}

		(void)MPI_Finalize();
	}

	static std::string mpi_thread_level_string(int level) {
		switch (level) {
		case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
		case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
		case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
		case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
		default: return "<unknown thread level>";
		}
	}

	virtual void init() {
		int r;

		// In an OpenMP program: no need for thread support
		int required;
		int provided;
		// required = MPI_THREAD_MULTIPLE;
		required = MPI_THREAD_SINGLE;
		r = MPI_Init_thread(NULL, NULL, required, &provided);
		mpi_error_test(r, "MPI_Init_thread fails");
		if (provided < required) {
			throw MCMCException("Cannot provide required MT level " +
							   	mpi_thread_level_string(required) +
							   	", get " + mpi_thread_level_string(provided));
		}

		r = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
		mpi_error_test(r, "MPI_Comm_set_errhandler fails");

		r = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
		mpi_error_test(r, "MPI_Comm_size() fails");
		r = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
		mpi_error_test(r, "MPI_Comm_rank() fails");

		r = MPI_Type_contiguous(2, MPI_INT, &MPI_INT2_TUPLE);
		mpi_error_test(r, "MPI_Type_contiguous() fails");
		r = MPI_Type_commit(&MPI_INT2_TUPLE);
		mpi_error_test(r, "MPI_Type_commit() fails");

		DKV::TYPE::DKV_TYPE dkv_type = DKV::TYPE::FILE;

		const std::vector<std::string> &a = args.getRemains();
		std::vector<std::string> dkv_args;
		for (::size_t i = 0; i < a.size(); i++) {
			if (false) {
			} else if (i < a.size() - 1 && a[i] == "--dkv:type") {
				i++;
				if (false) {
				} else if (a[i] == "file") {
					dkv_type = DKV::TYPE::FILE;
#ifdef ENABLE_RAMCLOUD
				} else if (a[i] == "ramcloud") {
					dkv_type = DKV::TYPE::RAMCLOUD;
#endif
#ifdef ENABLE_RDMA
				} else if (a[i] == "rdma") {
					dkv_type = DKV::TYPE::RDMA;
					dkv_args.push_back("rdma:mpi-initialized");
#endif
				} else {
					std::cerr << "Possible values for dkv:file:" << std::endl;
					std::cerr << "   " << "file" << std::endl;
#ifdef ENABLE_RAMCLOUD
					std::cerr << "   " << "ramcloud" << std::endl;
#endif
#ifdef ENABLE_RDMA
					std::cerr << "   " << "rdma" << std::endl;
#endif
					throw mcmc::InvalidArgumentException("Unknown value \"" + a[i] +
														 "\" for dkv:type option");
				}
			} else {
				dkv_args.push_back(a[i]);
			}
		}

		// d_kv_store = new DKV::DKVRamCloud::DKVStoreRamCloud();
		switch (dkv_type) {
		case DKV::TYPE::FILE:
			d_kv_store = new DKV::DKVFile::DKVStoreFile();
			break;
#ifdef ENABLE_RAMCLOUD
		case DKV::TYPE::RAMCLOUD:
			d_kv_store = new DKV::DKVRamCloud::DKVStoreRamCloud();
			break;
#endif
#ifdef ENABLE_RDMA
		case DKV::TYPE::RDMA:
			d_kv_store = new DKV::DKVRDMA::DKVStoreRDMA();
			break;
#endif
		}

		max_minibatch_nodes = network.minibatch_nodes_for_strategy(mini_batch_size, strategy);
		max_minibatch_neighbors = max_minibatch_nodes * (real_num_node_sample() + mpi_size_ - 1) / mpi_size_;
		std::cerr << "minibatch size param " << mini_batch_size << " max " << max_minibatch_nodes << " #neighbors(total) " << max_minibatch_neighbors << std::endl;

		d_kv_store->Init(K + 1,
						 N,
						 max_minibatch_nodes + max_minibatch_neighbors,
                         max_minibatch_nodes,
						 dkv_args);

		if (mpi_rank_ == mpi_master_) {
			init_beta();
		}

		// Make kernelRandom init depend on mpi_rank_
		delete kernelRandom;
		kernelRandom = new Random::Random((mpi_rank_ + 1) * 42);
		threadRandom_.resize(omp_get_max_threads());
		for (::size_t i = 0; i < threadRandom_.size(); i++) {
			threadRandom_[i] = new Random::Random(i, (mpi_rank_ + 1) * 42);
		}

		t_populate_pi.start();
		init_pi();
		t_populate_pi.stop();
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

		grads_.resize(omp_get_max_threads());
		for (auto & g : grads_) {
			g.resize(K * max_minibatch_nodes);
		}
		scratch_.resize(K * max_minibatch_nodes);

        std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
		std::cerr << "Done constructor" << std::endl;
	}


    virtual void run() {
        /** run mini-batch based MCMC sampler, based on the sungjin's note */

        using namespace std::chrono;

		std::vector<std::vector<double>> phi_node(max_minibatch_nodes, std::vector<double>(K + 1));
		std::vector<double *> pi_node(max_minibatch_nodes);
		std::vector<double *> pi_neighbor(max_minibatch_neighbors);

		int r;

		r = MPI_Barrier(MPI_COMM_WORLD);
		mpi_error_test(r, "MPI_Barrier(initial) fails");

		t_start = clock();
        while (step_count < max_iteration && ! is_converged()) {

			t_outer.start();
			auto l1 = std::chrono::system_clock::now();
			//if (step_count > 200000){
				//interval = 2;
			//}
			check_perplexity();

			t_broadcast_beta.start();
			r = MPI_Bcast(beta.data(), beta.size(), MPI_DOUBLE, mpi_master_, MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Bcast of beta fails");
			t_broadcast_beta.stop();

			t_deploy_minibatch.start();
			// Defines:
			//	my_work_item_[] transformed to indicate ranks i.s.o. node ids
			// 	nodes_[]	minibatch rank -> node id
			// 	neighbor_[]		local neighbor_ rank -> node id
			EdgeSample edgeSample = deploy_mini_batch();
			t_deploy_minibatch.stop();

			std::cerr << "FIXME: use an AllGather to broadcast the minibatch pi" << std::endl;
			// ************ load minibatch node pi from D-KV store **************
            t_load_pi_minibatch.start();
			pi_node.resize(nodes_.size());
            d_kv_store->ReadKVRecords(pi_node, nodes_, DKV::RW_MODE::READ_ONLY);
            t_load_pi_minibatch.stop();

			std::cerr << "FIXME: skip this if we just do locality, no load balancing" << std::endl;
			// ************ load neighor pi from D-KV store **********
			t_load_pi_neighbor.start();
			pi_neighbor.resize(neighbor_.size());
			d_kv_store->ReadKVRecords(pi_neighbor,
									  neighbor_,
									  DKV::RW_MODE::READ_ONLY);
			t_load_pi_neighbor.stop();

			double eps_t = a * std::pow(1 + step_count / b, -c);	// step size
			// double eps_t = std::pow(1024+step_count, -0.5);

			// Calculate grads in parallel, partitioned over the neighbors
			t_grads_calc.start();
			::size_t num_threads;
			num_threads = grads_calculate(pi_node, pi_neighbor, eps_t);
			t_grads_calc.stop();

			// Locally add the grads contribution of each thread
			t_grads_sum_local.start();
			// Sum to grads_[thread = 0]
			grads_sum_local(num_threads);
			t_grads_sum_local.stop();

			// Reduce(+) the grads to their owner
			t_grads_reduce.start();
			allreduce2all();
			t_grads_reduce.stop();

			std::cerr << "FIXME: every node must update its OWN minibatch phi" << std::endl;
			t_update_phi.start();
			update_phi(constify(pi_node), eps_t, &phi_node);
			t_update_phi.stop();

			// all synchronize with barrier: ensure we read pi/phi_sum from current iteration
			t_barrier_phi.start();
			r = MPI_Barrier(MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Barrier(post phi) fails");
			t_barrier_phi.stop();

			// TODO calculate and store updated values for pi/phi_sum
			std::cerr << "FIXME: every node must update its OWN minibatch pi" << std::endl;
			t_update_pi.start();
#pragma omp parallel for // num_threads (12)
			for (::size_t i = 0; i < pi_node.size(); i++) {
				pi_from_phi(pi_node[i], phi_node[i]);
			}
			t_update_pi.stop();

			// std::cerr << "write back phi/pi after update" << std::endl;
			std::cerr << "FIXME: every node must store its OWN minibatch pi" << std::endl;
			t_store_pi_minibatch.start();
			d_kv_store->WriteKVRecords(nodes_, constify(pi_node));
			t_store_pi_minibatch.stop();

			t_barrier_pi.start();
			r = MPI_Barrier(MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Barrier(post pi) fails");
			t_barrier_pi.stop();

			if (mpi_rank_ == mpi_master_) {
				t_update_beta.start();
				update_beta(*edgeSample.first, pi_node, eps_t, edgeSample.second);
				t_update_beta.stop();

				// TODO FIXME allocate this outside the loop
				delete edgeSample.first;
			}

			d_kv_store->PurgeKVRecords();

            step_count++;
			t_outer.stop();
			auto l2 = std::chrono::system_clock::now();
			if (false) {
				std::cout << "LOOP  = " << (l2-l1).count() << std::endl;
			}
		}

		check_perplexity();

		Timer::printHeader(std::cout);
		std::cout << t_populate_pi << std::endl;
		std::cout << t_outer << std::endl;
		std::cout << t_deploy_minibatch << std::endl;
		std::cout << t_mini_batch << std::endl;
		std::cout << t_nodes_in_mini_batch << std::endl;
		std::cout << t_sample_neighbor_nodes << std::endl;
		std::cout << t_load_pi_minibatch << std::endl;
		std::cout << t_load_pi_neighbor << std::endl;
		std::cout << t_grads_calc << std::endl;
		std::cout << t_grads_sum_local << std::endl;
		std::cout << t_grads_reduce << std::endl;
		std::cout << t_update_phi << std::endl;
		std::cout << t_barrier_phi << std::endl;
		std::cout << t_update_pi << std::endl;
		std::cout << t_store_pi_minibatch << std::endl;
		std::cout << t_purge_pi_perp << std::endl;
		std::cout << t_barrier_pi << std::endl;
		std::cout << t_update_beta << std::endl;
		std::cout << t_load_pi_beta << std::endl;
		std::cout << t_broadcast_beta << std::endl;
		std::cout << t_perplexity << std::endl;
		std::cout << t_load_pi_perp << std::endl;
		std::cout << t_rank_pi_perp << std::endl;
		std::cout << t_cal_edge_likelihood << std::endl;
		std::cout << t_perp_log << std::endl;
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
		// FIXME: might parallelize this, but it is startup-only
		for (int32_t i = mpi_rank_; i < static_cast<int32_t>(N); i += mpi_size_) {
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
			// Easy performance improvement: accumulate records up to write area
			// size, the Write/Purge
			d_kv_store->WriteKVRecords(node, pi_wrapper);
			d_kv_store->PurgeKVRecords();
		}
	}


	void check_perplexity() {
		if (mpi_rank_ == mpi_master_) {
			if (step_count % interval == 0) {
				t_perplexity.start();
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
					std::string file_name = "mcmc_stochastic_" + to_string (K) + "_num_nodes_" + to_string(num_node_sample) + "_us_air.txt";
					myfile.open (file_name);
					int size = ppxs_held_out.size();
					for (int i = 0; i < size; i++){

						//int iteration = i * 100 + 1;
						myfile <<iterations[i]<<"    "<<timings[i]<<"    "<<ppxs_held_out[i]<<"\n";
					}

					myfile.close();
				}
			}
		}

		//print "step: " + str(self._step_count)
		/**
		  pr = cProfile.Profile()
		  pr.enable()
		  */
	}


	EdgeSample deploy_mini_batch() {
		std::vector<int32_t> items(mpi_size_);		// used only at Master	FIXME: lift to class
		std::vector<WorkItem> scatter_items;			// used only at Master	FIXME: lift to class
		std::vector<int32_t> scatter_displs(mpi_size_);	// used only at Master	FIXME: lift to class
		int		r;
		EdgeSample edgeSample;

		if (mpi_rank_ == mpi_master_) {
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

			// Master samples the neighbors for each minibatch node.
			// The tuples <minibatch_node, neighbor(minibatch_node)> are distributed
			// over the workers, in accordance with the neighbor locality.
			t_nodes_in_mini_batch.start();
			OrderedVertexSet node_set = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();
// std::cerr << "mini_batch size " << mini_batch.size() << " num_node_sample " << num_node_sample << std::endl;

			std::cerr << "FIXME: do the neighbor bucketize also in parallel" << std::endl;
			std::cerr << "FIXME: do better load balancing; fetching a few pi rows won't kill you" << std::endl;

			nodes_.resize(node_set.size());
			std::copy(node_set.begin(), node_set.end(), nodes_.begin());
			std::vector<NeighborSet> neighbor(nodes_.size());
			std::cerr << "FIXME: distribute the neighbor sampling?" << std::endl;
			t_sample_neighbor_nodes.start();
#pragma omp parallel for
			for (::size_t i = 0; i < nodes_.size(); ++i) {
				neighbor[i] = sample_neighbor_nodes(num_node_sample, nodes_[i],
													threadRandom_[omp_get_thread_num()]);
			}
			t_sample_neighbor_nodes.stop();
			std::cerr << "FIXME: static allocation of work_items[]" << std::endl;
			std::cerr << "FIXME: do two passes over work_items to save a copy" << std::endl;
			std::vector<std::vector<WorkItem>> work_item(mpi_size_);
			for (::size_t i = 0; i < nodes_.size(); ++i) {
				for (auto j : neighbor[i]) {
					// ::size_t h = d_kv_store->HostOf(j);
					::size_t h = (j * mpi_size_) / N;
					work_item[h].push_back(WorkItem(i, j));
				}
			}

			scatter_items.clear();
			int32_t running_sum = 0;
			for (int i = 0; i < mpi_size_; i++) {
				scatter_displs[i] = running_sum;
				items[i] = work_item[i].size();
				running_sum += items[i];
				scatter_items.insert(scatter_items.end(),
									 work_item[i].begin(),
									 work_item[i].begin() + items[i]);
			}
		}

		int32_t nw;
		r = MPI_Scatter(items.data(), 1, MPI_INT,
						&nw, 1, MPI_INT,
						mpi_master_, MPI_COMM_WORLD);
		mpi_error_test(r, "MPI_Scatter of minibatch chunks fails");
		num_my_work_items_ = nw;
		my_work_item_.resize(num_my_work_items_);

		if (mpi_rank_ == mpi_master_) {
			// TODO Master scatters the <minibatch, neigbor> tuples over the workers,
			// preferably with consideration for both load balance and locality
			r = MPI_Scatterv(scatter_items.data(),
							 items.data(),
							 scatter_displs.data(),
							 MPI_INT2_TUPLE,
							 my_work_item_.data(),
							 num_my_work_items_,
							 MPI_INT2_TUPLE,
							 mpi_master_,
							 MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Scatterv of work tuples fails at master");

		} else {
			r = MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
							 my_work_item_.data(), num_my_work_items_, MPI_INT,
							 mpi_master_, MPI_COMM_WORLD);
			mpi_error_test(r, "MPI_Scatterv of work tuples fails at worker");
		}

		if (false) {
			std::cerr << "Master gives me work items [" << num_my_work_items_ << "] ";
			for (auto n : my_work_item_) {
				std::cerr << n << " ";
			}
			std::cerr << std::endl;
		}

		uint32_t minibatch_size = nodes_.size();
		r = MPI_Bcast(&minibatch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		mpi_error_test(r, "MPI_Bcast of minibatch size fails");
		nodes_.resize(minibatch_size);
		r = MPI_Bcast(nodes_.data(), nodes_.size(), MPI_INT, 0, MPI_COMM_WORLD);
		mpi_error_test(r, "MPI_Bcast of minibatch nodes fails");

		std::cerr << "Master gives me minibatch nodes[" << minibatch_size << "] ";
		for (auto n : nodes_) {
			std::cerr << n << " ";
		}
		std::cerr << std::endl;

		convert_to_ranking();

		return edgeSample;
	}


	void convert_to_ranking() {
#ifdef MASTER_GIVES_ME_NODE_ISO_RANK
		std::unordered_map<int32_t, int32_t> ranking;
		for (::size_t i = 0; i < nodes_.size(); i++) {
			ranking[nodes_[i]] = i;
		}
#endif
		neighbor_.resize(my_work_item_.size());
#pragma omp parallel for	// allow concurrent write access to neighbor_, my_work_item_
		for (::size_t i = 0; i < my_work_item_.size(); ++i) {
			neighbor_[i] = my_work_item_[i].neighbor_node;
			my_work_item_[i].neighbor_node = i;
#ifdef MASTER_GIVES_ME_NODE_ISO_RANK
			my_work_item_[i].minibatch_node = ranking[my_work_item_[i].minibatch_node];
#endif
		}
	}


	::size_t grads_calculate(const std::vector<double *> &pi_node,
							 const std::vector<double *> &pi_neighbor,
							 const double eps_t) {
		assert(grads_.capacity() == static_cast<::size_t>(omp_get_max_threads()));
#pragma omp parallel for
		for (::size_t t = 0; t < grads_.size(); t++) {
			// std::cerr << "FIXME: do this lazily:" << std::endl;
			// grads_[t].reserve(K * nodes_.size());
			assert(grads_[t].size() >= nodes_.size());
			for (::size_t i = 0; i < nodes_.size(); ++i) {
				for (::size_t k = 0; k < K; k++) {
					grads_[t][i * K + k] = 0.0;
				}
			}
		}

#ifndef NDEBUG
		// The _node fields are in fact ranks (indices) in the vectors
		for (::size_t i = 0; i < my_work_item_.size(); ++i) {
			assert((::size_t)my_work_item_[i].minibatch_node < pi_node.size());
			assert((::size_t)my_work_item_[i].neighbor_node < pi_neighbor.size());
		}
#endif

		::size_t num_threads;
// #pragma omp parallel for private(my_pi_node, my_pi_neighbor, my_grads, probs)
#pragma omp parallel for
		for (::size_t i = 0; i < my_work_item_.size(); ++i) {
			if (i == 0 && omp_get_thread_num() == 0) {
				num_threads = omp_get_num_threads();
			}
			int32_t node_rank = my_work_item_[i].minibatch_node;
			int32_t neighbor_rank = my_work_item_[i].neighbor_node;

			if ((::size_t)node_rank >= pi_node.size()) {
				std::cerr << "Oops, node_rank " << node_rank << " exceeds " << pi_node.size() << std::endl;
			}
			if ((::size_t)neighbor_rank >= pi_neighbor.size()) {
				std::cerr << "Oops, neighbor_rank " << neighbor_rank << " exceeds " << pi_neighbor.size() << std::endl;
			}

			const double *my_pi_node = pi_node[node_rank];
			const double *my_pi_neighbor = pi_neighbor[neighbor_rank];
			if (false) {
				std::cerr << __func__ << " ";
				std::cerr << "phi[" << i << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << (my_pi_node[k] * my_pi_node[K]) << " ";
				}
				std::cerr << std::endl;
				std::cerr << "pi[" << node_rank << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << my_pi_node[k] << " ";
				}
				std::cerr << std::endl;
				std::cerr << "pi[" << neighbor_rank << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << my_pi_neighbor[k] << " ";
				}
				std::cerr << std::endl;
			}

			int32_t node = nodes_[node_rank];
			int32_t neighb = neighbor_[neighbor_rank];
			if (node != neighb) {
				int y_ab = 0;		// observation
				Edge edge(std::min(node, neighb), std::max(node, neighb));
				if (edge.in(network.get_linked_edges())) {
					y_ab = 1;
				}

				if (true) {
					static bool first = true;
					if (first) {
						first = false;
						std::cerr << "FIXME: statically allocate probs[num_threads][K]" << std::endl;
					}
				}
				std::vector<double> probs(K);
				double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
				for (::size_t k = 0; k < K; k++) {
					double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
					// FIXME: first multiply by my_pi_node[k]
					probs[k] = my_pi_node[k] * (my_pi_neighbor[k] * f + e);
					// probs[k] = my_pi_neighbor[k] * f + e;
				}

				// double prob_sum = np::inner_product(probs.data(), my_pi_node,
													// probs.size(), 0.0);
				double prob_sum = np::sum(probs);
				double phi_i_sum = my_pi_node[K];
				// std::cerr << std::fixed << std::setprecision(12) << "node " << node << " neighb " << neighb << " prob_sum " << prob_sum << " phi_i_sum " << phi_i_sum << " #sample " << real_num_node_sample() << std::endl;

				double *my_grads = &grads_[omp_get_thread_num()][node_rank * K];
				for (::size_t k = 0; k < K; k++) {
					// grads_[k] += (probs[k] / prob_sum) / phi[node_index][k] - 1.0 / phi_i_sum;
					// FIXME: then divide by my_pi_node[k]
					my_grads[k] += ((probs[k] / prob_sum) / my_pi_node[k] - 1.0) / phi_i_sum;
					// my_grads[k] += (probs[k] / prob_sum - 1.0) / phi_i_sum;
				}
			}
		}

		return num_threads;
    }


	void grads_sum_local(::size_t num_threads) {
#pragma omp parallel for
		for (::size_t i = 0; i < grads_.size(); ++i) {
			for (::size_t t = 1; t < num_threads; ++t) {
				grads_[0][i] += grads_[t][i];
			}
		}
	}


	void allreduce2all() {
		std::cerr << "FIXME: implement this allreduce2all" << std::endl;
	}


    void update_phi(const std::vector<const double *> &pi_node,
                    const double eps_t,
					std::vector<std::vector<double>> *phi_node	// out parameter
					) {
		const double Nn = (1.0 * N) / num_node_sample;
#pragma omp parallel for
		for (::size_t i = 0; i < pi_node.size(); ++i) {
			// std::cerr << "Random seed " << std::hex << "0x" << kernelRandom->seed(0) << ",0x" << kernelRandom->seed(1) << std::endl << std::dec;
			// update_phi(i, pi_node[i], grads_.begin() + i * K,
			// 		   eps_t, threadRandom_[i], &phi_node[i]);
			const double *my_pi_node = pi_node[i];
			// grads_[0] contains the reduced sum of the sum of the local thread contributions
			const double *my_grads = &grads_[0][i * K];
			std::vector<double> &my_phi_node = (*phi_node)[i];
			Random::Random *rnd = threadRandom_[omp_get_thread_num()];

			std::vector<double> noise = rnd->randn(K);	// random gaussian noise.
			if (false) {
				for (::size_t k = 0; k < K; ++k) {
					std::cerr << "randn " << std::fixed << std::setprecision(12) << noise[k] << std::endl;
				}
			}
			double phi_i_sum = my_pi_node[K];
			// update phi for node i
			for (::size_t k = 0; k < K; k++) {
				double phi_node_k = my_pi_node[k] * phi_i_sum;
				my_phi_node[k] = std::abs(phi_node_k + eps_t / 2 * (alpha - phi_node_k + \
																	Nn * my_grads[k]) +
										  sqrt(eps_t * phi_node_k) * noise[k]);
			}

			if (false) {
				std::cerr << std::fixed << std::setprecision(12) << __func__ << " post Nn " << Nn << " phi[" << nodes_[i] << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << my_phi_node[k] << " ";
				}
				std::cerr << std::endl;
				std::cerr << "pi[" << nodes_[i] << "] ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << my_pi_node[k] << " ";
				}
				std::cerr << std::endl;
				std::cerr << "grads_ ";
				for (::size_t k = 0; k < K; k++) {
					std::cerr << std::fixed << std::setprecision(12) << my_grads[k] << " ";
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
	}


    void update_beta(const OrderedEdgeSet &mini_batch,
					 const std::vector<double *> &pi_node,
					 const double eps_t,
					 const double scale) {

		std::cerr << "FIXME: allocate update_beta::grads statically (or share with update_phi)" << std::endl;
		std::vector<std::vector<std::vector<double> > > grads(omp_get_max_threads(), std::vector<std::vector<double> >(K, std::vector<double>(2, 0.0)));	// gradients K*2 dimension
        // sums = np.sum(self.__theta,1)
		std::vector<double> theta_sum(theta.size());
		std::transform(theta.begin(), theta.end(), theta_sum.begin(), np::sum<double>);

		// FIXME: already did the nodes_in_batch() -- only the ranking remains
		std::unordered_map<int, int> node_rank;
		for (::size_t i = 0; i < nodes_.size(); i++) {
			node_rank[nodes_[i]] = i;
		}

		// update gamma, only update node in the grad
		//double eps_t = std::pow(1024+step_count, -0.5);
		// for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
        std::vector<Edge> v_mini_batch(mini_batch.begin(), mini_batch.end());
#pragma omp parallel for // num_threads (12)
        for (::size_t e = 0; e < v_mini_batch.size(); e++) {
			std::vector<double> probs(K);
            const auto *edge = &v_mini_batch[e];

            int y = 0;
            if (edge->in(network.get_linked_edges())) {
                y = 1;
			}
			int i = node_rank[edge->first];
			int j = node_rank[edge->second];

			double pi_sum = 0.0;
			for (::size_t k = 0; k < K; k++) {
				// Note: this is the KV-store cached pi, not the Learner item
				pi_sum += pi_node[i][k] * pi_node[j][k];
				double f = pi_node[i][k] * pi_node[j][k];
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
				grads[omp_get_thread_num()][k][0] += f * ((1 - y) / theta[k][0] - one_over_theta_sum);
				grads[omp_get_thread_num()][k][1] += f * (y / theta[k][1] - one_over_theta_sum);
			}
		}

#pragma omp parallel for
		for (int i = 1; i < omp_get_max_threads(); i++) {
			for (::size_t k = 0; k < K; k++) {
				grads[0][k][0] += grads[i][k][0];
				grads[0][k][1] += grads[i][k][1];
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
													  scale * grads[0][k][i]) +
									   f * noise[k][i]);
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
		std::cerr << "FIXME: isn't the held-out set fixed across iterations?" << std::endl;
        // FIXME isn't the held-out set fixed across iterations
        // so we can memo the ranking?
		t_rank_pi_perp.start();
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
		t_rank_pi_perp.stop();

		t_load_pi_perp.start();
		std::vector<double *> pi(node_rank.size());
		d_kv_store->ReadKVRecords(pi, nodes, DKV::RW_MODE::READ_ONLY);
		t_load_pi_perp.stop();

		std::cerr <<  "FIXME: OpenMP-parallelize calc perplexity" << std::endl;
        // FIXME: trivial to OpenMP-parallelize: memo the likelihoods, sum
        // them in the end
		::size_t i = 0;
		for (auto edge : data) {
			const Edge &e = edge.first;
			int a = node_rank[e.first];
			int b = node_rank[e.second];
			t_cal_edge_likelihood.start();
			double edge_likelihood = cal_edge_likelihood(pi[a], pi[b],
														 edge.second, beta);
			t_cal_edge_likelihood.stop();
			if (std::isnan(edge_likelihood)) {
				std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
			}

			//cout<<"AVERAGE COUNT: " <<average_count;
			ppx_for_heldout[i] = (ppx_for_heldout[i] * (average_count-1) + edge_likelihood)/(average_count);
			// std::cout << std::fixed << std::setprecision(12) << e << " in? " << (e.in(network.get_linked_edges()) ? "True" : "False") << " -> " << edge_likelihood << " av. " << average_count << " ppx[" << i << "] " << ppx_for_heldout[i] << std::endl;
			// assert(edge->second == e.in(network.get_linked_edges()));
            t_perp_log.start();
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
            t_perp_log.stop();
			i++;
		}

		t_purge_pi_perp.start();
		d_kv_store->PurgeKVRecords();
		t_purge_pi_perp.stop();

		// std::cout << std::setprecision(12) << "ratio " << link_ratio << " count: link " << link_count << " " << link_likelihood << " non-link " << non_link_count << " " << non_link_likelihood << std::endl;

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
			std::cout << std::fixed << std::setprecision(12) << avg_likelihood << " " << (link_likelihood / link_count) << " " << link_count << " " << \
				(non_link_likelihood / non_link_count) << " " << non_link_count << " " << avg_likelihood1 << std::endl;
			// std::cout << "perplexity score is: " << exp(-avg_likelihood) << std::endl;
		}

		// return std::exp(-avg_likelihood);


		//if (step_count > 1000000)
		average_count = average_count + 1;
		std::cout << "average_count is: " << average_count << " ";
		return (-avg_likelihood);
	}


	int node_owner(int node) const {
		return node % mpi_size_;
	}


	static void mpi_error_test(int r, const std::string &message) {
		if (r != MPI_SUCCESS) {
			throw MCMCException("MPI error " + r + message);
		}
	}


protected:
	::size_t	max_minibatch_nodes;
	::size_t	max_minibatch_neighbors;

	const int mpi_master_;
	int		mpi_size_;
	int		mpi_rank_;
	MPI_Datatype MPI_INT2_TUPLE;

	const Options &args;

	DKV::DKVStoreInterface *d_kv_store;

	uint64_t num_my_work_items_;
	std::vector<WorkItem> my_work_item_;	// [] tuple<minibatch rank, neighbor_ rank>
	std::vector<int32_t> nodes_;			// rank -> id of minibatch node
	std::vector<int32_t> neighbor_;			// rank -> id of neighbor_ node

	std::vector<std::vector<double>> grads_;
	std::vector<double> scratch_;

	std::vector<Random::Random *> threadRandom_;

	Timer t_outer;
	Timer t_populate_pi;
	Timer t_perplexity;
	Timer t_rank_pi_perp;
	Timer t_cal_edge_likelihood;
	Timer t_perp_log;
	Timer t_mini_batch;
	Timer t_nodes_in_mini_batch;
	Timer t_sample_neighbor_nodes;
	Timer t_grads_calc;
	Timer t_grads_sum_local;
	Timer t_grads_reduce;
	Timer t_update_phi;
	Timer t_update_pi;
	Timer t_update_beta;
	Timer t_load_pi_beta;
	Timer t_load_pi_minibatch;
	Timer t_load_pi_neighbor;
	Timer t_load_pi_perp;
	Timer t_store_pi_minibatch;
	Timer t_purge_pi_perp;
	Timer t_broadcast_beta;
	Timer t_deploy_minibatch;
   	Timer t_barrier_phi;
	Timer t_barrier_pi;

	clock_t	t_start;
	std::vector<double> timings;
};

}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
