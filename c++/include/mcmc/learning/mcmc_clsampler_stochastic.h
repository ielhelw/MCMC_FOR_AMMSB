#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>
#include <chrono>

#include <boost/config.hpp>

#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reductor.hpp>

#include "opencl/context.h"

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

#define do_stringify(str)	#str
#define stringify(str)		do_stringify(str)
#define ERROR_MESSAGE_LENGTH (4 * 1024)


class MCMCClSamplerStochastic : public Learner {


	class GraphWrapper {
	public:
		// TODO: share buffers for clNodes clEdges

		void init(MCMCClSamplerStochastic &learner, cl::Buffer *clGraph, const std::map<int, std::vector<int> > &edges, ::size_t N, ::size_t num_nodes_in_batch, ::size_t num_edges_in_batch) {
			this->clGraph = clGraph;
			this->num_nodes_in_batch = num_nodes_in_batch;
			this->num_edges_in_batch = num_edges_in_batch;

			adjacency_list.resize(N);
			for (auto e: edges) {
				adjacency_list[e.first] = e.second;	// This is a full copy, right?
				// Question: is there a point in sorting the adjacency lists?
				std::sort(adjacency_list[e.first].begin(), adjacency_list[e.first].end());
				if (false && (e.first == 1218 || find(e.second.begin(), e.second.end(), 1218) != e.second.end())) {
					std::cerr << "Edge init: found [" << e.first;
					for (auto n: e.second) {
						std::cerr << " " << n;
					}
					std::cerr << "]" << std::endl;
				}
			}

			*clGraph = learner.createBuffer("graph", CL_MEM_READ_WRITE,
					2*64/8 /* 2 pointers, each is at most 64-bits */
					);
			clNodes = learner.createBuffer("clNodes", CL_MEM_READ_ONLY,
					num_nodes_in_batch * sizeof h_subNodes[0]
					);
			clEdges = learner.createBuffer("clEdges", CL_MEM_READ_ONLY,
					num_edges_in_batch * sizeof h_subEdges[0]
					);

			learner.graph_init_kernel.setArg(0, *clGraph);
			learner.graph_init_kernel.setArg(1, clEdges);
			learner.graph_init_kernel.setArg(2, clNodes);

			learner.clContext.queue.enqueueTask(learner.graph_init_kernel);
			learner.clContext.queue.finish();
		}

		std::vector<std::vector<int> >	adjacency_list;

		cl::Buffer						*clGraph;
		cl::Buffer						clNodes;
		cl::Buffer						clEdges;

		std::vector<cl_int2>			h_subNodes;
		std::vector<cl_int>				h_subEdges;

		::size_t						num_edges_in_batch;
		::size_t						num_nodes_in_batch;
	};


public:
	MCMCClSamplerStochastic(const Options &args, const Network &graph, cl::ClContext clContext)
		: Learner(args, graph), clContext(clContext),
		  vexContext(std::vector<cl::Context>(1, clContext.getContext()),
					 std::vector<cl::CommandQueue>(1, clContext.getQueue())),
		  csumDouble(vexContext), csumInt(vexContext),
   		  kernelRandom(42),
		  groupSize(args.openclGroupSize), numGroups(args.openclNumGroups),
		  globalThreads(args.openclGroupSize * args.openclNumGroups),
		  kRoundedThreads(round_up_to_multiples(K, groupSize)), totalAllocedClBufers(0) {

#ifdef RANDOM_FOLLOWS_CPP
		std::cout << "RANDOM_FOLLOWS_CPP enabled" << std::endl;
#endif
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		std::cout << "Randomness IS compatible with nonscaling graph version" << std::endl;
#else
		std::cout << "Randomness is NOT compatible with nonscaling graph version" << std::endl;
#endif

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning
		// num_node_sample = N / 5;
		if (args.num_node_sample == 0) {
			// num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
			num_node_sample = N / 50;
		} else {
			num_node_sample = args.num_node_sample;
		}
		if (args.interval == 0) {
			interval = 50;
		} else {
			interval = args.interval;
		}
		if (args.mini_batch_size == 0) {
			mini_batch_size = N / 10;	// old default for STRATIFIED_RANDOM_NODE_SAMPLING
		}
		std::cerr << "num_node_sample " << num_node_sample << " a " << a << " b " << b << " c " << c << " alpha " << alpha << " eta (" << eta[0] << "," << eta[1] << ")" << std::endl;

		info(std::cout);

		t_outer = timer::Timer("  outer");
		t_perplexity = timer::Timer("  perplexity");
		t_mini_batch = timer::Timer("  sample_mini_batch");
		t_nodes_in_mini_batch = timer::Timer("  nodes_in_mini_batch");
		t_stage_held_out = timer::Timer("  stage_held_out");
		t_sample_neighbor_nodes = timer::Timer("  sample_neighbor_nodes");
		t_stage_graph = timer::Timer("  stage_graph");
		t_stage_pi = timer::Timer("  stage_pi");
		t_update_phi = timer::Timer("  update_phi");
		t_stage_pi_neighbors = timer::Timer("      stage_pi_neighbors");
		t_stage_pi_neighbors_gather = timer::Timer("      stage_pi_neighbors_gather");
		total_data_stage_pi_neighbors = 0;
		t_stage_phi = timer::Timer("  stage_phi");
		t_update_pi = timer::Timer("  update_pi");
		t_update_beta = timer::Timer("  update_beta");

		t_kernel_neighbors = timer::Timer("    kernel_neighbors");
		t_kernel_update_phi = timer::Timer("    kernel_update_phi");
		t_kernel_update_pi = timer::Timer("    kernel_update_pi");
		t_kernel_calculate_beta = timer::Timer("    kernel_calculate_beta");
		t_kernel_calculate_theta = timer::Timer("    kernel_calculate_theta");
		t_kernel_calculate_grads = timer::Timer("    kernel_calculate_grads");

		hash_table_size = round_next_power_2(real_num_node_sample());
		if ((double)hash_table_size / real_num_node_sample() < 1.8) {
			hash_table_size *= 2;
		}

		std::ostringstream opts;
		opts << "-I" << BOOST_STRINGIZE(PROJECT_HOME) << "/../OpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << real_num_node_sample()
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N
			 << " -DRAND_MAX=" << std::numeric_limits<uint64_t>::max() << "ULL"
#ifdef NDEBUG
			 << " -DNDEBUG"
#endif
#ifdef RANDOM_FOLLOWS_CPP
			 << " -DRANDOM_FOLLOWS_CPP"
#endif
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
			 << " -DRANDOM_FOLLOWS_SCALABLE_GRAPH"
#endif
			 << " -DHASH_SIZE=" << hash_table_size;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cerr << "FIXME FIXME FIXME redo max. edges for strategy" << std::endl;
		::size_t num_edges_in_batch = network.minibatch_edges_for_strategy(mini_batch_size, strategy);
		::size_t num_nodes_in_batch = network.minibatch_nodes_for_strategy(mini_batch_size, strategy);

		const ::size_t deviceMem = clContext.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		bufferSize = args.openclBufferSize;
		if (bufferSize == 0) {
			bufferSize = deviceMem;
		}

		// Sub-batching to calculate z_ab. Necessary because not all of pi/phi fits
		// in device memory. Batch by dividing neighbors.
		subNeighbors = real_num_node_sample();
		::size_t nodes_neighbors = num_nodes_in_batch * (1 + subNeighbors);
		if (K * nodes_neighbors * sizeof(cl_double) > bufferSize) {
			if (bufferSize / (K * sizeof(cl_double) * num_nodes_in_batch) < 2) {
				throw BufferSizeException("Buffer too small for nodes + 1 neighbor chunk");
			}
			subNeighbors = bufferSize / (K * sizeof(cl_double) * num_nodes_in_batch) - 1;
			std::cout << "pi/phi submatrix does not fit in device. Limit neighbor subset size to " << subNeighbors << std::endl;
			nodes_neighbors = num_nodes_in_batch * (1 + subNeighbors);
		}
		std::cerr << "OpenCL group size " << groupSize << " num " << numGroups << " buffer size " << bufferSize << " neighbor subset size " << subNeighbors << std::endl;

		hostPiBuffer = std::vector<double>(nodes_neighbors * K);
		hostPhiBuffer = std::vector<double>(nodes_neighbors * K);
		hostNeighbors = std::vector<cl_int>(num_nodes_in_batch * real_num_node_sample());

		graph_program = this->clContext.createProgram(BOOST_STRINGIZE(PROJECT_HOME) "/../OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");

		init_graph();

		sampler_program = this->clContext.createProgram(BOOST_STRINGIZE(PROJECT_HOME) "/../OpenCL/sampler.cl", progOpts);
		random_gamma_kernel = cl::Kernel(sampler_program, "random_gamma");
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		random_gamma_dummy_kernel = cl::Kernel(sampler_program, "random_gamma_dummy");
#endif
		row_normalize_kernel = cl::Kernel(sampler_program, "row_normalize");
		sample_neighbor_nodes_kernel = cl::Kernel(sampler_program, "sample_neighbor_nodes");
		update_phi_kernel = cl::Kernel(sampler_program, "update_phi");
		update_pi_kernel = cl::Kernel(sampler_program, "update_pi");
		update_beta_calculate_theta_sum_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta_sum");
		update_beta_calculate_grads_kernel = cl::Kernel(sampler_program, "update_beta_calculate_grads");
		update_beta_calculate_theta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta");
		update_beta_calculate_beta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_beta");
		cal_perplexity_kernel = cl::Kernel(sampler_program, "cal_perplexity");
		init_buffers_kernel = cl::Kernel(sampler_program, "init_buffers");

		clBuffers = createBuffer("clBuffers", CL_MEM_READ_WRITE, 64 * 100); // enough space for 100 * 8 pointers

		clNodes = createBuffer("clNodes", CL_MEM_READ_ONLY,
				num_nodes_in_batch * sizeof(cl_int)
				);
		clNodesNeighbors = createBuffer("clNodesNeighbors", CL_MEM_READ_WRITE,
				num_nodes_in_batch * real_num_node_sample() * sizeof(cl_int)
				);
		clNodesNeighborsHash = createBuffer("clNodesNeighborsHash", CL_MEM_READ_WRITE,
				num_nodes_in_batch * hash_table_size * sizeof(cl_int)
				);

		clEdges = createBuffer("clEdges", CL_MEM_READ_ONLY,
				num_edges_in_batch * sizeof(cl_int2)
				);

		clPi = createBuffer("clPi", CL_MEM_READ_WRITE,
				nodes_neighbors * K * sizeof(cl_double) // #total_nodes x #K
				);
		clPhi = createBuffer("clPhi", CL_MEM_READ_WRITE,
				num_nodes_in_batch * K * sizeof(cl_double) // #total_nodes x #K
				);

		clBeta = createBuffer("clBeta", CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #K
				);
		clTheta = createBuffer("clTheta", CL_MEM_READ_WRITE,
				2 * K * sizeof(cl_double) // 2 x #K
				);
		clThetaSum = createBuffer("clThetaSum", CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #K
				);

		clScratch = createBuffer("clScratch", CL_MEM_READ_WRITE,
				std::max(num_nodes_in_batch, globalThreads) * K * (sizeof(cl_double2) + sizeof(cl_double))
				);
		clRandomSeed = createBuffer("clRandomSeed", CL_MEM_READ_WRITE,
				globalThreads * sizeof(cl_ulong2)
				);
		clErrorCtrl = createBuffer("clErrorCtrl", CL_MEM_READ_WRITE,
				sizeof(cl_int16)
				);
		clErrorMsg = createBuffer("clErrorMsg", CL_MEM_READ_WRITE,
				ERROR_MESSAGE_LENGTH
				);
		clStoredRandom = createBuffer("clStoredRandom", CL_MEM_READ_WRITE,
									  num_nodes_in_batch * K * sizeof(cl_double));

		int Idx = 0;
		init_buffers_kernel.setArg(Idx++, clBuffers);
		init_buffers_kernel.setArg(Idx++, clGraph);
		init_buffers_kernel.setArg(Idx++, clHeldOutGraph);
		init_buffers_kernel.setArg(Idx++, clNodes);
		init_buffers_kernel.setArg(Idx++, clNodesNeighbors);
		init_buffers_kernel.setArg(Idx++, clNodesNeighborsHash);
		init_buffers_kernel.setArg(Idx++, clEdges);
		init_buffers_kernel.setArg(Idx++, clPi);
		init_buffers_kernel.setArg(Idx++, clPhi);
		std::cerr << "Phi size " << num_nodes_in_batch << " * " << K << " = " << (num_nodes_in_batch * K) << std::endl;
		init_buffers_kernel.setArg(Idx++, clBeta);
		init_buffers_kernel.setArg(Idx++, clTheta);
		init_buffers_kernel.setArg(Idx++, clThetaSum);
		init_buffers_kernel.setArg(Idx++, clScratch);
		init_buffers_kernel.setArg(Idx++, clRandomSeed);
		init_buffers_kernel.setArg(Idx++, clErrorCtrl);
		init_buffers_kernel.setArg(Idx++, clErrorMsg);
		init_buffers_kernel.setArg(Idx++, clStoredRandom);
		try {
			clContext.queue.enqueueTask(init_buffers_kernel);
		} catch (cl::Error &e) {
			if (e.err() == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
				std::cerr << "Failed to allocate CL buffers (sum = " << (double)totalAllocedClBufers/(1024*1024) << " MB)" << std::endl;
				for (auto b : clBufAllocSizes) {
					std::cerr << "Buffer: " << b.first << " = " << (double)b.second/(1024*1024) << " MB" << std::endl;
				}
			}
			throw e;
		}

		std::vector<cl_ulong2> randomSeed(globalThreads);
		for (unsigned int i = 0; i < randomSeed.size(); ++i) {
			randomSeed[i].s[0] = 42 + i;
			randomSeed[i].s[1] = 42 + i + 1;
		}
		clContext.queue.enqueueWriteBuffer(clRandomSeed, CL_TRUE,
				0, randomSeed.size() * sizeof(cl_ulong2),
				randomSeed.data());

		// theta = Random::random->gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
		// // std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
		// // theta = Random::random->gamma(100.0, 0.01, K, 2);		// parameterization for \beta
		::size_t K_workers = std::min(K, static_cast< ::size_t>(globalThreads));
		Idx = 0;
		random_gamma_kernel.setArg(Idx++, clBuffers);
		random_gamma_kernel.setArg(Idx++, clTheta);
		random_gamma_kernel.setArg(Idx++, (cl_double)eta[0]);
		random_gamma_kernel.setArg(Idx++, (cl_double)eta[1]);
		random_gamma_kernel.setArg(Idx++, (cl_int)K);
		random_gamma_kernel.setArg(Idx++, 2);
		clContext.queue.enqueueNDRangeKernel(random_gamma_kernel, cl::NullRange, cl::NDRange(K_workers), cl::NDRange(1));

#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		// Advance the random state, so device random and host kernelRandom are in sync
		// (void)kernelRandom.gamma(100.0, 0.01, K, 2);		// parameterization for \beta
		clContext.queue.finish();
		std::cerr << "Grab another K " << K << " host dummy randoms" << std::endl;
		(void)kernelRandom.gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
		std::cerr << "Done K = " << K << " host dummy randoms" << std::endl;
#endif

        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		// phi = Random::random->gamma(1, 1, N, K);					// parameterization for \pi
		phi = kernelRandom.gamma(1, 1, N, K);					// parameterization for \pi
		std::cerr << "Done host random for phi" << std::endl;
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);

#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		// Advance the device random state, so device random and host kernelRandom are in sync
		clContext.queue.finish();
		std::cerr << "Grab another N * K = " << (N * K) << " device dummy randoms" << std::endl;
		Idx = 0;
		random_gamma_dummy_kernel.setArg(Idx++, clBuffers);
		random_gamma_dummy_kernel.setArg(Idx++, (double)1.0);
		random_gamma_dummy_kernel.setArg(Idx++, (double)1.0);
		random_gamma_dummy_kernel.setArg(Idx++, (int)N);
		random_gamma_dummy_kernel.setArg(Idx++, (int)K);
		clContext.queue.enqueueNDRangeKernel(random_gamma_dummy_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(1));
		clContext.queue.finish();
		std::cerr << "Done N * K = " << (N * K) << " device dummy randoms" << std::endl;
#endif

#if 0
		for (unsigned i = 0; i < N; ++i) {
			// copy phi
			clContext.queue.enqueueWriteBuffer(clPhi, CL_TRUE,
					i * K * sizeof(cl_double),
					K * sizeof(cl_double),
					phi[i].data());
			// Copy pi
			clContext.queue.enqueueWriteBuffer(clPi, CL_TRUE,
					i * K * sizeof(double),
					K * sizeof(double),
					pi[i].data());
		}
#endif

		if (false) {
			std::vector<std::vector<double> > theta(K, std::vector<double>(2));

			for (::size_t i = 0; i < K; ++i) {
				// copy phi
				clContext.queue.enqueueReadBuffer(clTheta, CL_TRUE,
												  i * 2 * sizeof(cl_double),
												  2 * sizeof(cl_double),
												  theta[i].data());
			}
			std::cout << "theta[*][0]: ";
			for (::size_t k = 0; k < K; k++) {
				std::cout << theta[k][0] << " ";
			}
			std::cout << std::endl;
			std::cout << "theta[*][1]: ";
			for (::size_t k = 0; k < K; k++) {
				std::cout << theta[k][1] << " ";
			}
			std::cout << std::endl;

#ifdef INITIALIZE_PHI_ON_DEVICE
			std::vector<std::vector<double> > pi(N, std::vector<double>(K));            // parameterization for \pi
			std::vector<std::vector<double> > phi(N, std::vector<double>(K));           // parameterization for \pi

			for (::size_t i = 0; i < N; ++i) {
				// copy phi
				clContext.queue.enqueueReadBuffer(clPhi, CL_TRUE,
												  i * K * sizeof(cl_double),
												  K * sizeof(cl_double),
												  phi[i].data());
				// Copy pi
				clContext.queue.enqueueReadBuffer(clPi, CL_TRUE,
												  i * K * sizeof(double),
												  K * sizeof(double),
												  pi[i].data());
			}
#endif

			for (::size_t i = 0; i < 10; i++) {
				std::cerr << "phi[" << i << "]: ";
				for (::size_t k = 0; k < 10; k++) {
					std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
				}
				std::cerr << std::endl;
			}

			for (::size_t i = 0; i < 10; i++) {
				std::cerr << "pi[" << i << "]: ";
				for (::size_t k = 0; k < 10; k++) {
					std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
				}
				std::cerr << std::endl;
			}
		}

        // // temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        // // self._beta = temp[:,1]
		// std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		// np::row_normalize(&temp, theta);
		// std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));

		t_kernel_calculate_beta.start();
		Idx = 0;
		update_beta_calculate_beta_kernel.setArg(Idx++, clBuffers);
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_beta_kernel, cl::NullRange, cl::NDRange(K_workers), cl::NDRange(1));

		clContext.queue.finish();
		t_kernel_calculate_beta.stop();
		clContext.queue.enqueueReadBuffer(clBeta, CL_FALSE, 0, K * sizeof(cl_double), beta.data());
		if (false) {
			clContext.queue.finish();
			std::cerr << "beta ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << beta[k] << " ";
			}
			std::cerr << std::endl;
		}

		clContext.queue.finish();

#if 0
		// copy theta
		std::vector<cl_double2> vclTheta(theta.size());
		std::transform(theta.begin(), theta.end(), vclTheta.begin(), [](const std::vector<double>& in){
			cl_double2 ret;
			ret.s[0] = in[0];
			ret.s[1] = in[1];
			return ret;
		});
		clContext.queue.enqueueWriteBuffer(clTheta, CL_TRUE,
				0, theta.size() * sizeof(cl_double2),
				vclTheta.data());

		// copy beta
		clContext.queue.enqueueWriteBuffer(clBeta, CL_TRUE, 0, K * sizeof(double), beta.data());
#endif

		// fill Hash with EMPTY (-1) values
		vex::backend::opencl::device_vector<cl_int> vexDeviceHash(clNodesNeighborsHash);
		vex::vector<cl_int> vexHash(vexContext.queue(0), vexDeviceHash);
		vexHash = (cl_int) -1;

		vex::backend::opencl::device_vector<cl_int> vexDeviceErr(clErrorCtrl);
		vex::vector<cl_int> vexErr(vexContext.queue(0), vexDeviceErr);
		vexErr = 0;

		errMsg = new char[ERROR_MESSAGE_LENGTH];

		clLinkLikelihood = createBuffer("clLinkLikelihood", CL_MEM_READ_WRITE, globalThreads * sizeof(cl_double));
		clNonLinkLikelihood = createBuffer("clNonLinkLikelihood", CL_MEM_READ_WRITE, globalThreads * sizeof(cl_double));
		clLinkCount = createBuffer("clLinkCount", CL_MEM_READ_WRITE, globalThreads * sizeof(cl_int));
		clNonLinkCount = createBuffer("clNonLinkCount", CL_MEM_READ_WRITE, globalThreads * sizeof(cl_int));

		clContext.queue.finish();

		std::cerr << "Done constructor" << std::endl;
	}

	virtual ~MCMCClSamplerStochastic() {
	}

	virtual void run() {
		/** run mini-batch based MCMC sampler, based on the sungjin's note */
		timer::Timer::setTabular(true);

		while (step_count < max_iteration && ! is_converged()) {

			// auto l1 = std::chrono::system_clock::now();
			t_outer.start();

			if (step_count % interval == 0) {
				t_perplexity.start();
				double ppx_score = cal_perplexity_held_out();
				t_perplexity.stop();
				std::cout << std::fixed << std::setprecision(12) << "step count: " << step_count << " perplexity for hold out set: " << ppx_score << std::endl;
				ppxs_held_out.push_back(ppx_score);
			}

			// (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			// std::cerr << "Invoke sample_mini_batch" << std::endl;
			t_mini_batch.start();
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy);
			t_mini_batch.stop();
			// std::cerr << "Done sample_mini_batch" << std::endl;
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			if (mini_batch.size() == 0) {
				std::cerr << "Empty mini batch; who cares nowadays?" << std::endl;
			}

			// iterate through each node in the mini batch.
			t_nodes_in_mini_batch.start();
			const std::vector<int> nodes = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();

			// std::cerr << "Stage clSubHeldOutGraph" << std::endl;
			t_stage_held_out.start();
			stage_subgraph(&clSubHeldOutGraph, nodes);
			t_stage_held_out.stop();

			// std::cerr << "Sample neighbor nodes" << std::endl;
			t_sample_neighbor_nodes.start();
			sample_neighbor_nodes(nodes);
			t_sample_neighbor_nodes.stop();

			// std::cerr << "Stage clSubGraph" << std::endl;
			t_stage_graph.start();
			stage_subgraph(&clSubGraph, nodes);
			t_stage_graph.stop();

			// Pi is staged in two contiguous sections:
			// 1) pi(i) for i in nodes
			// 2) pi(n'(i)) for n'(i): the subset [offset:size] of neighbors n(i) of node i
			t_stage_pi.start();
			stage_sub_vectors(clPi, pi, hostPiBuffer, nodes);
			t_stage_pi.stop();
			t_stage_phi.start();
			stage_sub_vectors(clPhi, phi, hostPhiBuffer, nodes);
			clContext.queue.finish();
			t_stage_phi.stop();

			if (false) {
				std::unordered_set<int> unique;
				for (auto i = hostNeighbors.begin(); i < hostNeighbors.begin() + nodes.size() * real_num_node_sample(); i++) {
					unique.insert(*i);
				}
				std::cerr << "Neighbors " << (nodes.size() * real_num_node_sample()) << " unique " << unique.size() << " fraction " << (unique.size() / (1.0 * nodes.size() * real_num_node_sample())) << std::endl;
			}
			if (false) {
				for (::size_t i = 0; i < nodes.size(); i++) {
					std::cerr << "Neigbors[" << i << "] ";
					for (::size_t n = 0; n < real_num_node_sample(); n++) {
						std::cerr << hostNeighbors[i * real_num_node_sample() + n] << " ";
					}
					std::cerr << std::endl;
				}
			}

			::size_t offset = 0;
			while (offset < real_num_node_sample()) {
				::size_t size = subNeighbors;
				if (offset + size > real_num_node_sample()) {
					size = real_num_node_sample() - offset;
				}

				t_update_phi.start();
				update_phi(nodes, offset, subNeighbors);
				t_update_phi.stop();

				offset += size;
			}

			t_update_pi.start();
			update_pi(nodes);
			commit_phi(nodes);
			t_update_pi.stop();

			// sample (z_ab, z_ba) for each edge in the mini_batch.
			// z is map structure. i.e  z = {(1,10):3, (2,4):-1}
			if (false) {
				std::cerr << "Nodes in mini_batch: ";
				for (auto n: nodes) {
					std::cerr << n << " ";
				}
				std::cerr << std::endl;
				std::cerr << "Edges in mini_batch: ";
				for (auto e: mini_batch) {
					std::cerr << e << " ";
				}
				std::cerr << std::endl;
			}

			t_update_beta.start();
			update_beta(mini_batch, scale, nodes);
			t_update_beta.stop();

			delete edgeSample.first;

			step_count++;
			t_outer.stop();
			// auto l2 = std::chrono::system_clock::now();
			// std::cout << "LOOP  = " << (l2-l1).count() << std::endl;
		}

		timer::Timer::printHeader(std::cout);
		std::cout << t_outer << std::endl;
		std::cout << t_perplexity << std::endl;
		std::cout << t_mini_batch << std::endl;
		std::cout << t_nodes_in_mini_batch << std::endl;
		std::cout << t_stage_held_out << std::endl;
		std::cout << t_sample_neighbor_nodes << std::endl;
		std::cout << t_stage_graph << std::endl;
		std::cout << t_stage_pi << std::endl;
		std::cout << t_update_phi << std::endl;
		std::cout << t_stage_pi_neighbors << std::endl;
		std::cout << t_stage_pi_neighbors_gather << std::endl;
		double mb = total_data_stage_pi_neighbors / (1.0 * (1 << 20));
		std::cout << "    Wrote staged data " << mb << "MB " <<
			"time " << std::chrono::duration_cast<std::chrono::milliseconds>(t_stage_pi_neighbors_gather.total()).count() << "ms " <<
			"throughput " << (mb / std::chrono::duration_cast<std::chrono::milliseconds>(t_stage_pi_neighbors_gather.total()).count()) << "GB/s" << std::endl;
		std::cout << t_stage_phi << std::endl;
		std::cout << t_update_pi << std::endl;
		std::cout << t_update_beta << std::endl;
		std::cout << t_kernel_calculate_beta << std::endl;
		std::cout << t_kernel_calculate_theta << std::endl;
		std::cout << t_kernel_calculate_grads << std::endl;
		std::cout << t_kernel_neighbors << std::endl;
		std::cout << t_kernel_update_phi << std::endl;
		std::cout << t_kernel_update_pi << std::endl;
	}

protected:

#if defined INITIALIZE_PHI_ON_DEVICE || defined RANDOM_FOLLOWS_SCALABLE_GRAPH
	void device_row_normalize(cl::Buffer &clPi, cl::Buffer &clPhi, ::size_t N, ::size_t K) {
		// pi.resize(phi.size(), std::vector<double>(phi[0].size()));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		// pi[N,K] = row_normalize(phi[N,K])
		// s[i] = sum_j phi[i,j]
		// pi[i,j] = phi[i,j] / s[i]
		int arg;

		arg = 0;
		row_normalize_kernel.setArg(arg++, clPhi);
		row_normalize_kernel.setArg(arg++, clPi);
		row_normalize_kernel.setArg(arg++, (cl_int)N);
		row_normalize_kernel.setArg(arg++, (cl_int)K);

		clContext.queue.enqueueNDRangeKernel(row_normalize_kernel,
											 cl::NullRange,
											 cl::NDRange(globalThreads),
											 cl::NDRange(1));
		clContext.queue.finish();
	}
#endif

	void stage_edges(const OrderedEdgeSet &mini_batch, const std::vector<int> &nodes) {
		::size_t i;
		std::vector<cl_int2> edges(mini_batch.size());
		std::unordered_map<int, int> node_index(N);
		i = 0;
		for (auto n: nodes) {
			node_index[n] = i;
			i++;
		}
		i = 0;
		for (auto e: mini_batch) {
			edges[i].s[0] = node_index[e.first];
			edges[i].s[1] = node_index[e.second];
			i++;
		}
		if (false) {
			std::cerr << "Edges: ";
			for (auto e: edges) {
				std::cerr << e << " ";
			}
			std::cerr << std::endl;
		}

		clContext.queue.enqueueWriteBuffer(clEdges, CL_TRUE,
				0, edges.size()*sizeof(cl_int2),
				edges.data());
	}

	void update_beta(const OrderedEdgeSet &mini_batch, double scale, const std::vector<int> &nodes) {
		int arg;

		// We assume that are staged correctly:
		//  - linked_edges graph G
		//  - pi(nodes(mini_batch))

		arg = 0;
		update_beta_calculate_theta_sum_kernel.setArg(arg++, clTheta);
		update_beta_calculate_theta_sum_kernel.setArg(arg++, clThetaSum);

		t_kernel_calculate_theta.start();
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_theta_sum_kernel,
											 cl::NullRange, cl::NDRange(kRoundedThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		t_kernel_calculate_theta.stop();

		if (nodes.size() != 0) {
			stage_edges(mini_batch, nodes);
		}

		::size_t countPartialSums = std::min(mini_batch.size(), globalThreads);
		::size_t calcGradsThreads = round_up_to_multiples(countPartialSums, groupSize);

		cl::Event e_grads_kernel;
		if (nodes.size() != 0) {
			arg = 0;
			update_beta_calculate_grads_kernel.setArg(arg++, clBuffers);
			update_beta_calculate_grads_kernel.setArg(arg++, (cl_int)mini_batch.size());
			update_beta_calculate_grads_kernel.setArg(arg++, (cl_double)scale);
			update_beta_calculate_grads_kernel.setArg(arg++, (cl_int)countPartialSums);
			update_beta_calculate_grads_kernel.setArg(arg++, (cl_double)epsilon);	// move to clBuffers

			clContext.queue.finish(); // Wait for previous kernel, staging of edges

			t_kernel_calculate_grads.start();
			clContext.queue.enqueueNDRangeKernel(update_beta_calculate_grads_kernel, cl::NullRange,
					cl::NDRange(calcGradsThreads), cl::NDRange(groupSize),
					NULL, &e_grads_kernel);
		}

		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		cl_double2 clEta;
		clEta.s[0] = this->eta[0];
		clEta.s[1] = this->eta[1];

		if (nodes.size() != 0) {
			e_grads_kernel.wait();
			t_kernel_calculate_grads.stop();
		}

		arg = 0;
		update_beta_calculate_theta_kernel.setArg(arg++, clBuffers);
		update_beta_calculate_theta_kernel.setArg(arg++, (cl_double)scale);
		update_beta_calculate_theta_kernel.setArg(arg++, (cl_double)eps_t);
		update_beta_calculate_theta_kernel.setArg(arg++, clEta);
		update_beta_calculate_theta_kernel.setArg(arg++, (int)countPartialSums);

		clContext.queue.enqueueTask(update_beta_calculate_theta_kernel);
		clContext.queue.finish();

		if (false) {
			std::vector<cl_double2> theta(K);
			clContext.queue.enqueueReadBuffer(clTheta, CL_TRUE, 0, K * sizeof(cl_double2), theta.data());
			std::cerr << __func__ << std::endl;
			std::cerr << "theta ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << "(" << theta[k].s[0] << "," << theta[k].s[1] << ") ";
			}
			std::cerr << std::endl;
		}

		arg = 0;
		update_beta_calculate_beta_kernel.setArg(arg++, clBuffers);

		t_kernel_calculate_beta.start();
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_beta_kernel, cl::NullRange, cl::NDRange(kRoundedThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		t_kernel_calculate_beta.stop();

		clContext.queue.enqueueReadBuffer(clBeta, CL_FALSE, 0, K * sizeof(cl_double), beta.data());
		clContext.queue.finish();
		if (false) {
			std::cerr << __func__ << std::endl;
			std::cerr << "beta ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << beta[k] << " ";
			}
			std::cerr << std::endl;
		}
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}

	void stage_subgraph(GraphWrapper *graph, const std::vector<int> &nodes) {
		// Copy held_out_set subgraph for nodes
		// ::size_t edge_elts = nodes.size() * std::max(network.get_max_fan_out(), num_node_sample + 1);
		if (nodes.size() == 0) {
			return;
		}

		graph->h_subEdges.clear();
		graph->h_subNodes.clear();
		graph->h_subNodes.resize(nodes.size());
		::size_t i = 0;
		::size_t offset = 0;
		for (auto node: nodes) {
			const auto &neighbors = graph->adjacency_list[node];
			graph->h_subEdges.insert(graph->h_subEdges.end(), neighbors.begin(), neighbors.end());
			graph->h_subNodes[i].s[0] = neighbors.size();
			graph->h_subNodes[i].s[1] = offset;
			i++;
			offset += neighbors.size();
			if (false && (node == 1218 || node == 918)) {
				std::cerr << "Insert node " << node << " neighbors [";
				for (::size_t i = offset - neighbors.size(); i < offset; i++) {
					std::cerr << graph->h_subEdges[i] << " ";
				}
				std::cerr << "] offset becomes " << offset << std::endl;
			}
		}

		if (graph->h_subEdges.size() == 0) {
			// std::cerr << "Non-empty minibatch but no neighbors" << std::endl;
			return;
		}

		assert(graph->h_subNodes.size() <= graph->num_nodes_in_batch);
		assert(graph->h_subEdges.size() <= graph->num_edges_in_batch);
		assert(sizeof graph->h_subNodes[0] == sizeof(cl_int2));
		assert(sizeof graph->h_subEdges[0] == sizeof(cl_int));

		clContext.queue.finish();
		clContext.queue.enqueueWriteBuffer(graph->clNodes, CL_FALSE, 0, graph->h_subNodes.size()*sizeof graph->h_subNodes[0], graph->h_subNodes.data());
		clContext.queue.enqueueWriteBuffer(graph->clEdges, CL_FALSE, 0, graph->h_subEdges.size()*sizeof graph->h_subEdges[0], graph->h_subEdges.data());
		clContext.queue.finish();
	}

	void stage_sub_vectors(cl::Buffer &buffer,
						   const std::vector<std::vector<double> > &data,
						   std::vector<double> &hostBuffer,
						   const std::vector<int> &nodes) {
		static int first = 1;
		if (first) {
			first = 0;
			std::cerr << "FIXME: " << __func__ << "(): ensure neighbor indirect at both host and device" << std::endl;
			std::cerr << "FIXME: " << __func__ << "(): implement subrange of matrix" << std::endl;
			std::cerr << "FIXME: " << __func__ << "(): if this is mapped memory: no need for intermediate copy" << std::endl;
		}

		if (nodes.size() == 0) {
			return;
		}

		hostBuffer.clear();
		for (auto n: nodes) {
			assert(data[n].size() == K);
			hostBuffer.insert(hostBuffer.end(), data[n].begin(), data[n].end());
#ifndef NDEBUG
			for (::size_t k = 0; k < K; k++) {
				assert(! isnan(data[n][k]));
			}
#endif
		}

		clContext.queue.enqueueWriteBuffer(buffer, CL_FALSE, 0, nodes.size() * K * sizeof(cl_double), hostBuffer.data());
	}

	/*
	 * Strided layout:
	 *  - neighbors is flattened vector of neighbors, real_num_node_sample() per node in the minibatch
	 *  - then offset is the offset into the first chunk of neighbors (so, per node)
	 *  - stride is the size of the current subset of neighbors per node
	 *  - size is real_num_node_sample()
	 * So to gather the data in this subset:
	 *     for each node
	 *         push pi for neighbor[offset .. offset + stride]
	 * and these pi lie contiguously in device memory, after the pi for the nodes themselves.
	 */
	void stage_sub_neighbor_vectors(cl::Buffer &buffer,
									const std::vector<std::vector<double> > &data,
									std::vector<double> &hostBuffer,
									const std::vector<int> &neighbors,
									::size_t mini_batch_size,
									::size_t offset, ::size_t stride, ::size_t size,
									::size_t bufferOffset) {
		static int first = 1;
		if (first) {
			first = 0;
			std::cerr << "FIXME: " << __func__ << "(): ensure neighbor indirect at both host and device" << std::endl;
			std::cerr << "FIXME: " << __func__ << "(): implement subrange of matrix" << std::endl;
			std::cerr << "FIXME: " << __func__ << "(): if this is mapped memory: no need for intermediate copy" << std::endl;
		}

		if (mini_batch_size == 0) {
			return;
		}

		::size_t n = std::min(size - offset, stride);

		if (false) {
			for (auto i = neighbors.begin() + offset; i < neighbors.begin() + mini_batch_size * size; i += size) {
				std::cerr << "Neighbors: ";
				for (auto b = i; b < i + n; b++) {
					std::cerr << *b << " ";
				}
				std::cerr << std::endl;
			}
		}

		hostBuffer.clear();
		t_stage_pi_neighbors_gather.start();
		for (auto i = neighbors.begin() + offset; i < neighbors.begin() + mini_batch_size * size; i += size) {
			for (auto b = i; b < i + n; b++) {
				assert(data[*b].size() == K);
				// t_stage_pi_neighbors_gather.start();
				hostBuffer.insert(hostBuffer.end(), data[*b].begin(), data[*b].end());
				// t_stage_pi_neighbors_gather.stop();
				if (false) {
					std::cerr << "pi_b: ";
					for (auto k: data[*b]) {
						std::cerr << std::fixed << std::setprecision(12) << k << " ";
					}
					std::cerr << std::endl;
				}
			}
		}
		t_stage_pi_neighbors_gather.stop();

		total_data_stage_pi_neighbors += mini_batch_size * n * K * sizeof(cl_double);

		assert(hostBuffer.size() == mini_batch_size * n * K);
		// std::cerr << "Stage hostBuffer size " << hostBuffer.size() << " must be " << (n * K) << std::endl;
		clContext.queue.enqueueWriteBuffer(buffer, CL_FALSE,
										   bufferOffset,
										   mini_batch_size * n * K * sizeof(cl_double),
										   hostBuffer.data());
		clContext.queue.finish();
	}

	void retrieve_sub_vectors(cl::Buffer &buffer,
							  std::vector<double> &hostBuffer,
							  ::size_t offset, ::size_t size) {
		clContext.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size * K * sizeof(cl_double), hostBuffer.data() + offset);
	}

	void sample_neighbor_nodes(const std::vector<int> &nodes) {

		if (nodes.size() == 0) {
			return;
		}

		clContext.queue.enqueueWriteBuffer(clNodes, CL_FALSE, 0, nodes.size() * sizeof(cl_int), nodes.data());

		int Idx = 0;
		sample_neighbor_nodes_kernel.setArg(Idx++, clBuffers);
		sample_neighbor_nodes_kernel.setArg(Idx++, (cl_int)nodes.size());

		clContext.queue.finish();
		t_kernel_neighbors.start();
		clContext.queue.enqueueNDRangeKernel(sample_neighbor_nodes_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		t_kernel_neighbors.stop();

		clContext.queue.enqueueReadBuffer(clNodesNeighbors, CL_FALSE, 0, nodes.size() * real_num_node_sample() * sizeof(cl_int), hostNeighbors.data());

		static int first = 1;
		if (first) {
			first = 0;
			std::cerr << "FIXME FIXME FIXME: shuffle/indirect/restage neighbor table" << std::endl;
		}
	}

	void update_phi(const std::vector<int> &nodes, ::size_t offset, ::size_t stride) {

		// Pi is staged in two contiguous sections:
		// 1) pi(i) for i in nodes, before the loop that calls us
		// 2) pi(n'(i)) for n'(i): the subset [offset:size] of neighbors n(i) of node i

		t_stage_pi_neighbors.start();
		stage_sub_neighbor_vectors(clPi, pi, hostPiBuffer, hostNeighbors,
								   nodes.size(),							// mini_batch_size
								   offset,									// offset
								   stride,									// stride
								   real_num_node_sample(),					// size = numNeighbors
								   nodes.size() * K * sizeof(cl_double));	// bufferOffset
		t_stage_pi_neighbors.stop();

		cl_double eps_t = a * std::pow(1 + step_count / b, -c);   // step size
		int Idx = 0;
		::size_t n = std::min(real_num_node_sample() - offset, stride);
		update_phi_kernel.setArg(Idx++, clBuffers);
		update_phi_kernel.setArg(Idx++, (cl_int)offset);
		update_phi_kernel.setArg(Idx++, (cl_int)n);
		update_phi_kernel.setArg(Idx++, (cl_int)nodes.size());
		// TODO FIXME put alpha, a, b, c, epsilon into global device struct
		update_phi_kernel.setArg(Idx++, (cl_double)alpha);
		update_phi_kernel.setArg(Idx++, (cl_double)epsilon);
		update_phi_kernel.setArg(Idx++, eps_t);

		clContext.queue.finish();
		check_for_kernel_errors(); // sample_neighbor_nodes_kernel
		t_kernel_update_phi.start();
		clContext.queue.enqueueNDRangeKernel(update_phi_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		t_kernel_update_phi.stop();
		check_for_kernel_errors(); // update_phi_kernel
	}

	void update_pi(const std::vector<int> &nodes) {
		if (nodes.size() == 0) {
			return;
		}

		int Idx = 0;
		update_pi_kernel.setArg(Idx++, clBuffers);
		update_pi_kernel.setArg(Idx++, (cl_int)nodes.size());

		clContext.queue.finish();
		check_for_kernel_errors(); // previous kernel
		t_kernel_update_pi.start();
		clContext.queue.enqueueNDRangeKernel(update_pi_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		t_kernel_update_pi.stop();
#if 0	// pi stays on the device
		// read Pi again
		for (auto node = nodes.begin();
				node != nodes.end();
				++node) {
			clContext.queue.enqueueReadBuffer(clPi, CL_FALSE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					pi[*node].data());
		}
#endif
		retrieve_sub_vectors(clPi, hostPiBuffer, 0, nodes.size());
		retrieve_sub_vectors(clPhi, hostPhiBuffer, 0, nodes.size());
	}

	void commit_phi(const std::vector<int> &nodes) {
		::size_t i = 0;
		for (auto n: nodes) {
			pi[n] = std::vector<cl_double>(hostPiBuffer.begin() + i * K, hostPiBuffer.begin() + (i + 1) * K);
			phi[n] = std::vector<cl_double>(hostPhiBuffer.begin() + i * K, hostPhiBuffer.begin() + (i + 1) * K);
			i++;
		}
	}

	std::vector<int> nodes_in_batch(const OrderedEdgeSet &mini_batch) const {
        /**
        Get all the unique nodes in the mini_batch.
         */
        OrderedVertexSet node_set;
        for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
            node_set.insert(edge->first);
            node_set.insert(edge->second);
		}

        return std::vector<int>(node_set.begin(), node_set.end());
	}

	double deviceSumDouble(cl::Buffer &xBuffer) {
		vex::backend::opencl::device_vector<double> xDev(xBuffer);
		vex::vector<double> X(vexContext.queue(0), xDev);

		double y = csumDouble(X);

		return y;
	}

	int deviceSumInt(cl::Buffer &xBuffer) {
		vex::backend::opencl::device_vector<int> xDev(xBuffer);
		vex::vector<int> X(vexContext.queue(0), xDev);

		int y = csumInt(X);

		return y;
	}

	template <typename T>
	T deviceSum(cl::Buffer &xBuffer) {
		vex::backend::opencl::device_vector<T> xDev(xBuffer);
		vex::vector<T> X(vexContext.queue(0), xDev);

		vex::Reductor<T, vex::SUM_Kahan> csum;
		T y = csum(X);

		return y;
	}

	double device_cal_perplexity_held_out() {
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
		std::cerr << "FIXME: chunk up pi/phi to incrementally calculate the likelihoods and counts" << std::endl;

		cl_double link_likelihood = 0.0;
		cl_double non_link_likelihood = 0.0;
		cl_int link_count = 0;
		cl_int non_link_count = 0;

		cl_int H = static_cast<cl_int>(network.get_held_out_set().size());

		int arg = 0;
		cal_perplexity_kernel.setArg(arg++, clIterableHeldOutGraph);
		cal_perplexity_kernel.setArg(arg++, H);
		cal_perplexity_kernel.setArg(arg++, clPi);
		cal_perplexity_kernel.setArg(arg++, clBeta);
		cal_perplexity_kernel.setArg(arg++, (cl_double)epsilon);
		cal_perplexity_kernel.setArg(arg++, clLinkLikelihood);
		cal_perplexity_kernel.setArg(arg++, clNonLinkLikelihood);
		cal_perplexity_kernel.setArg(arg++, clLinkCount);
		cal_perplexity_kernel.setArg(arg++, clNonLinkCount);

		clContext.queue.finish();
		clContext.queue.enqueueNDRangeKernel(cal_perplexity_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();

		link_likelihood = deviceSum<double>(clLinkLikelihood);
		non_link_likelihood = deviceSum<double>(clNonLinkLikelihood);
		link_count = deviceSum<int>(clLinkCount);
		non_link_count = deviceSum<int>(clNonLinkCount);

		clContext.queue.finish();
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
		return (-avg_likelihood);
	}

	void prepare_iterable_cl_graph(cl::Buffer &iterableGraph, const EdgeMap &data) {
		std::unique_ptr<cl_int3[]> hIterableGraph(new cl_int3[data.size()]);
		::size_t i = 0;
		for (auto a: data) {
			cl_int3 e;
			e.s[0] = a.first.first;
			e.s[1] = a.first.second;
			e.s[2] = a.second ? 1 : 0;
			hIterableGraph.get()[i] = e;
			i++;
		}
		iterableGraph = createBuffer("iterableGraph", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.size() * sizeof(cl_int3), hIterableGraph.get());
	}

	void init_graph() {
		graph_program = this->clContext.createProgram(BOOST_STRINGIZE(PROJECT_HOME) "/../OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");
		std::map<int, std::vector<int>> linkedMap;

		for (auto e : network.get_linked_edges()) {
			linkedMap[e.first].push_back(e.second);
			linkedMap[e.second].push_back(e.first);
		}

		::size_t num_edges_in_batch = network.minibatch_edges_for_strategy(mini_batch_size, strategy);
		::size_t num_nodes_in_batch = network.minibatch_nodes_for_strategy(mini_batch_size, strategy);
		num_edges_in_batch = std::max(num_edges_in_batch, network.get_max_fan_out(num_nodes_in_batch));
		clSubGraph.init(*this, &clGraph, linkedMap, N, num_nodes_in_batch, num_edges_in_batch);

		linkedMap.clear();

		for (auto e : network.get_held_out_set()) {
			linkedMap[e.first.first].push_back(e.first.second);
			linkedMap[e.first.second].push_back(e.first.first);
		}
		for (auto e : network.get_test_set()) {
			linkedMap[e.first.first].push_back(e.first.second);
			linkedMap[e.first.second].push_back(e.first.first);
		}
		clSubHeldOutGraph.init(*this, &clHeldOutGraph, linkedMap, N, num_nodes_in_batch, num_edges_in_batch);

		prepare_iterable_cl_graph(clIterableHeldOutGraph, network.get_held_out_set());

#if MCMC_CL_STOCHASTIC_TEST_GRAPH
		test_graph();
#endif
	}

#if MCMC_CL_STOCHASTIC_TEST_GRAPH // TEST GRAPH ON OPENCL's SIDE
	void test_graph() {
		cl::Program program2 = this->clContext.createProgram(BOOST_STRINGIZE(PROJECT_HOME) "/../OpenCL/test.cl", progOpts);
		cl::Kernel test_print_peers_of_kernel(program2, "test_print_peers_of");
		test_print_peers_of_kernel.setArg(0, clGraph);
		test_print_peers_of_kernel.setArg(1, 0);

		int n = 800;
		test_print_peers_of_kernel.setArg(1, n);

		std::cout << "LOOKING INTO NODE " << n << ", neighbors = " << linkedMap[n].size() << std::endl;
		for (auto p : linkedMap[n]) {
			std::cout << p << std::endl;
		}

		clContext.queue.enqueueTask(test_print_peers_of_kernel);
		clContext.queue.finish();
	}
#endif

	void check_for_kernel_errors() {
		cl_int err = 0;
		clContext.queue.enqueueReadBuffer(clErrorCtrl, CL_TRUE, 0, sizeof(cl_int), &err);
		if (err) {
			clContext.queue.enqueueReadBuffer(clErrorMsg, CL_TRUE, 0, ERROR_MESSAGE_LENGTH, errMsg);
			std::cerr << errMsg << std::endl;
			abort();
		}
	}

	/**
	 * returns the smallest #threads; #threads >= minRequired && #threads % groupSize == 0
	 */
	static inline ::size_t round_up_to_multiples(::size_t minRequired, ::size_t groupSize) {
		int numGroups = minRequired / groupSize;
		if (numGroups*groupSize < minRequired) {
			numGroups += 1;
		}
		return numGroups * groupSize;
	}

	cl::Buffer createBuffer(std::string name, cl_mem_flags flags, ::size_t size, void *ptr = NULL) {
		if (size > clContext.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) {
			std::stringstream s;
			s << "Cannot allocate buffer " << name << " of size " << size << " > " << clContext.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
			throw MCMCException(s.str());
		}
		cl::Buffer buf = cl::Buffer(clContext.context, flags, size, ptr);
		totalAllocedClBufers += size;
		clBufAllocSizes.push_back(std::make_pair(name, size));
		return buf;
	}

	static ::size_t round_next_power_2(::size_t v) {
		v -= 1;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v += 1;
		return v;
	}

	std::vector<double> hostPiBuffer;
	std::vector<double> hostPhiBuffer;
	std::vector<cl_int> hostNeighbors;
	::size_t bufferSize;
	::size_t subNeighbors;

	std::string progOpts;

	cl::ClContext clContext;

	cl::Program graph_program;
	cl::Program sampler_program;

	cl::Kernel graph_init_kernel;
	cl::Kernel random_gamma_kernel;
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
	cl::Kernel random_gamma_dummy_kernel;
#endif
	cl::Kernel row_normalize_kernel;
	cl::Kernel sample_neighbor_nodes_kernel;
	cl::Kernel update_phi_kernel;
	cl::Kernel update_pi_kernel;
	cl::Kernel update_beta_calculate_theta_sum_kernel;
	cl::Kernel update_beta_calculate_grads_kernel;
	cl::Kernel update_beta_calculate_theta_kernel;
	cl::Kernel update_beta_calculate_beta_kernel;
	cl::Kernel cal_perplexity_kernel;
	cl::Kernel init_buffers_kernel;

	cl::Buffer clBuffers;

	GraphWrapper clSubGraph;
	cl::Buffer clGraph;
	GraphWrapper clSubHeldOutGraph;
	cl::Buffer clHeldOutGraph;

	cl::Buffer clIterableHeldOutGraph;

	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clNodesNeighborsHash;
	cl::Buffer clEdges;
	cl::Buffer clPi;
	cl::Buffer clPhi;
	cl::Buffer clBeta;
	cl::Buffer clTheta;
	cl::Buffer clThetaSum;
	cl::Buffer clScratch;
	cl::Buffer clRandomSeed;
	cl::Buffer clErrorCtrl;
	cl::Buffer clErrorMsg;
	cl::Buffer clStoredRandom;

	::size_t hash_table_size;

	vex::Context vexContext;
	vex::Reductor<double, vex::SUM_Kahan> csumDouble;
	vex::Reductor<int, vex::SUM_Kahan> csumInt;

	cl::Buffer clLinkLikelihood;
	cl::Buffer clNonLinkLikelihood;
	cl::Buffer clLinkCount;
	cl::Buffer clNonLinkCount;

protected:
	// replicated in both mcmc_sampler_
	double	a;
	double	b;
	double	c;

	::size_t num_node_sample;
	::size_t interval;

	// To be deprecated:
	// std::vector<std::vector<double> > theta;		// parameterization for \beta
#ifndef INITIALIZE_PHI_ON_DEVICE
	std::vector<std::vector<double> > phi;			// parameterization for \pi
#endif
	Random::Random kernelRandom;

	const ::size_t groupSize;
	const ::size_t numGroups;
	const ::size_t globalThreads;
	const ::size_t kRoundedThreads;

	::size_t totalAllocedClBufers;
	std::vector<std::pair<std::string, ::size_t> > clBufAllocSizes;

	char *errMsg;

	timer::Timer t_outer;
	timer::Timer t_perplexity;
	timer::Timer t_mini_batch;
	timer::Timer t_nodes_in_mini_batch;
	timer::Timer t_sample_neighbor_nodes;
	timer::Timer t_update_phi;
	timer::Timer t_stage_held_out;
	timer::Timer t_stage_graph;
	timer::Timer t_stage_pi;
	timer::Timer t_stage_pi_neighbors;
	timer::Timer t_stage_pi_neighbors_gather;
	::size_t total_data_stage_pi_neighbors;
	timer::Timer t_stage_phi;
	timer::Timer t_update_pi;
	timer::Timer t_update_beta;
	timer::Timer t_kernel_calculate_beta;
	timer::Timer t_kernel_calculate_theta;
	timer::Timer t_kernel_calculate_grads;
	timer::Timer t_kernel_neighbors;
	timer::Timer t_kernel_update_phi;
	timer::Timer t_kernel_update_pi;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
