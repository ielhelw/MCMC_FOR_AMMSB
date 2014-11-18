#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>
#include <chrono>

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
		// FIXME FIXME FIXME What about NodesNeighbors and NodesNeighborsHash?????

		void init(MCMCClSamplerStochastic &learner, cl::Buffer *clGraph, const std::map<int, std::vector<int> > &edges, ::size_t N, ::size_t num_nodes_in_batch, ::size_t num_edges_in_batch) {
			this->clGraph = clGraph;
			this->num_nodes_in_batch = num_nodes_in_batch;
			this->num_edges_in_batch = num_edges_in_batch;

			adjacency_list.resize(N);
			for (auto e: edges) {
				adjacency_list[e.first] = e.second;	// This is a full copy, right?
				// Question: is there a point in sorting the adjacency lists?
				std::sort(adjacency_list[e.first].begin(), adjacency_list[e.first].end());
				if (e.first == 1218 || find(e.second.begin(), e.second.end(), 1218) != e.second.end()) {
					std::cerr << "Edge init: found [" << e.first;
					for (auto n: e.second) {
						std::cerr << " " << n;
					}
					std::cerr << "]" << std::endl;
				}
			}

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
		  globalThreads(groupSize * numGroups),
		  kRoundedThreads(round_up_to_multiples(K, groupSize)), totalAllocedClBufers(0) {

        // step size parameters.
        this->a = args.a;
        this->b = args.b;
        this->c = args.c;

        // control parameters for learning
		// num_node_sample = N / 5;
        num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
		std::cerr << "num_node_sample " << num_node_sample << std::endl;

		hash_table_size = round_next_power_2(real_num_node_sample());
		if ((double)hash_table_size / real_num_node_sample() < 1.8) {
			hash_table_size *= 2;
		}

		std::ostringstream opts;
		opts << "-I" << stringify(PROJECT_HOME) << "/../OpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << real_num_node_sample()
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N
			 << " -DRAND_MAX=" << std::numeric_limits<uint64_t>::max() << "ULL"
#ifdef RANDOM_FOLLOWS_CPP
			 << " -DRANDOM_FOLLOWS_CPP"
#endif
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
			 << " -DRANDOM_FOLLOWS_SCALABLE_GRAPH"
#endif
			 << " -DHASH_SIZE=" << hash_table_size;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cout << "num_node_sample = " << num_node_sample << std::endl;

		// BASED ON STRATIFIED RANDOM NODE SAMPLING STRATEGY
		::size_t num_edges_in_batch = N/10 + 1;
		::size_t num_nodes_in_batch = num_edges_in_batch + 1;

		::size_t nodes_neighbors = num_nodes_in_batch * (1 + real_num_node_sample());
		hostPiBuffer = std::vector<double>(nodes_neighbors * K);
		hostPhiBuffer = std::vector<double>(nodes_neighbors * K);
		hostNeighbors = std::vector<cl_int>(num_nodes_in_batch * real_num_node_sample());

		graph_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");

		init_graph();

		sampler_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/sampler.cl", progOpts);
		random_gamma_kernel = cl::Kernel(sampler_program, "random_gamma");
		row_normalize_kernel = cl::Kernel(sampler_program, "row_normalize");
		sample_latent_vars_neighbors_kernel = cl::Kernel(sampler_program, "sample_latent_vars_neighbors");
		sample_latent_vars_sample_z_ab_kernel = cl::Kernel(sampler_program, "sample_latent_vars_sample_z_ab");
		update_pi_kernel = cl::Kernel(sampler_program, "update_pi");
		sample_latent_vars2_kernel = cl::Kernel(sampler_program, "sample_latent_vars2");
		update_beta_calculate_theta_sum_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta_sum");
		update_beta_calculate_grads_kernel = cl::Kernel(sampler_program, "update_beta_calculate_grads");
		update_beta_calculate_theta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta");
		update_beta_calculate_beta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_beta");
#ifdef CALCULATE_PERPLEXITY_ON_DEVICE
		cal_perplexity_kernel = cl::Kernel(sampler_program, "cal_perplexity");
#endif
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
		// FIXME: make this nodes_neighbors * K
		clPi = createBuffer("clPi", CL_MEM_READ_WRITE,
				std::max(N, nodes_neighbors) * K * sizeof(cl_double) // #total_nodes x #K
				);
		// FIXME: make this nodes_neighbors * K
		clPhi = createBuffer("clPhi", CL_MEM_READ_WRITE,
				std::max(N, nodes_neighbors) * K * sizeof(cl_double) // #total_nodes x #K
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
		clZ = createBuffer("clZ", CL_MEM_READ_WRITE,
				num_nodes_in_batch * K * sizeof(cl_int)
				);
		clScratch = createBuffer("clScratch", CL_MEM_READ_WRITE,
				std::max(num_nodes_in_batch, globalThreads) * K * sizeof(cl_double)
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
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		clStoredRandom = createBuffer("clStoredRandom", CL_MEM_READ_WRITE,
									  num_nodes_in_batch * real_num_node_sample() * sizeof(cl_double));
#endif

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
		init_buffers_kernel.setArg(Idx++, clBeta);
		init_buffers_kernel.setArg(Idx++, clTheta);
		init_buffers_kernel.setArg(Idx++, clThetaSum);
		init_buffers_kernel.setArg(Idx++, clZ);
		init_buffers_kernel.setArg(Idx++, clScratch);
		init_buffers_kernel.setArg(Idx++, clRandomSeed);
		init_buffers_kernel.setArg(Idx++, clErrorCtrl);
		init_buffers_kernel.setArg(Idx++, clErrorMsg);
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		init_buffers_kernel.setArg(Idx++, clStoredRandom);
#endif
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

		// // theta = Random::random->gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
		// theta = Random::random->gamma(100.0, 0.01, K, 2);		// parameterization for \beta
		std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
		::size_t K_workers = std::min(K, static_cast< ::size_t>(globalThreads));
		Idx = 0;
		random_gamma_kernel.setArg(Idx++, clBuffers);
		random_gamma_kernel.setArg(Idx++, clTheta);
		random_gamma_kernel.setArg(Idx++, (double)100.0);
		random_gamma_kernel.setArg(Idx++, (double)0.01);
		random_gamma_kernel.setArg(Idx++, (int)K);
		random_gamma_kernel.setArg(Idx++, 2);
		clContext.queue.enqueueNDRangeKernel(random_gamma_kernel, cl::NullRange, cl::NDRange(K_workers), cl::NDRange(1));

#if defined CALCULATE_PHI_ON_DEVICE || defined RANDOM_FOLLOWS_SCALABLE_GRAPH
        // model parameters and re-parameterization
        // since the model parameter - \pi and \beta should stay in the simplex,
        // we need to restrict the sum of probability equals to 1.  The way we
        // restrict this is using re-reparameterization techniques, where we
        // introduce another set of variables, and update them first followed by
        // updating \pi and \beta.
		// phi = Random::random->gamma(1, 1, N, K);					// parameterization for \pi
		Idx = 0;
		random_gamma_kernel.setArg(Idx++, clBuffers);
		random_gamma_kernel.setArg(Idx++, clPhi);
		random_gamma_kernel.setArg(Idx++, (double)1.0);
		random_gamma_kernel.setArg(Idx++, (double)1.0);
		random_gamma_kernel.setArg(Idx++, (int)N);
		random_gamma_kernel.setArg(Idx++, (int)K);
		clContext.queue.enqueueNDRangeKernel(random_gamma_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(1));

		// pi.resize(phi.size(), std::vector<double>(phi[0].size()));
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		clContext.queue.finish();
		device_row_normalize(clPi, clPhi, N, K);
		// np::row_normalize(&pi, phi);
#endif
#ifndef CALCULATE_PHI_ON_DEVICE
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		(void)kernelRandom.gamma(100.0, 0.01, K, 2);		// parameterization for \beta
#endif
		phi = kernelRandom.gamma(1, 1, N, K);					// parameterization for \pi
        // self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
		pi.resize(phi.size(), std::vector<double>(phi[0].size()));
		np::row_normalize(&pi, phi);
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

		if (true) {
#ifdef CALCULATE_PHI_ON_DEVICE
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

		Idx = 0;
		update_beta_calculate_beta_kernel.setArg(Idx++, clTheta);
		update_beta_calculate_beta_kernel.setArg(Idx++, clBeta);
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_beta_kernel, cl::NullRange, cl::NDRange(K_workers), cl::NDRange(1));

#ifndef CALCULATE_PERPLEXITY_ON_DEVICE
		clContext.queue.finish();
		clContext.queue.enqueueReadBuffer(clBeta, CL_FALSE, 0, K * sizeof(double), beta.data());
#endif

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

		info(std::cout);
	}

	virtual ~MCMCClSamplerStochastic() {
	}

	virtual void run() {
		/** run mini-batch based MCMC sampler, based on the sungjin's note */
		timer::Timer t_outer("  outer");
		timer::Timer t_perplexity("  perplexity");
		timer::Timer t_mini_batch("  sample_mini_batch");
		timer::Timer t_nodes_in_mini_batch("  nodes_in_mini_batch");
		timer::Timer t_latent_vars("  sample_latent_vars");
		timer::Timer t_latent_vars2("  sample_latent_vars2");
		timer::Timer t_update_pi("  update_pi");
		timer::Timer t_update_beta("  update_beta");
		timer::Timer::setTabular(true);

		if (step_count % 1 == 0) {
			t_perplexity.start();
			double ppx_score = cal_perplexity_held_out();
			t_perplexity.stop();
			std::cout << std::fixed << std::setprecision(15) << "perplexity for hold out set is: " << ppx_score << std::endl;
			ppxs_held_out.push_back(ppx_score);
		}

		while (step_count < max_iteration && ! is_converged()) {
			auto l1 = std::chrono::system_clock::now();
			t_outer.start();

			// (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			t_mini_batch.start();
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
			t_mini_batch.stop();
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			// iterate through each node in the mini batch.
			t_nodes_in_mini_batch.start();
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);
			t_nodes_in_mini_batch.stop();

			t_latent_vars.start();
			sample_latent_vars(nodes);
			t_latent_vars.stop();

			t_update_pi.start();
			update_pi(nodes);
			t_update_pi.stop();

			// sample (z_ab, z_ba) for each edge in the mini_batch.
			// z is map structure. i.e  z = {(1,10):3, (2,4):-1}
			t_latent_vars2.start();
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

			sample_latent_vars2(mini_batch, nodes);
			t_latent_vars2.stop();

			t_update_beta.start();
			update_beta(mini_batch, scale);
			t_update_beta.stop();

			if (step_count % 1 == 0) {
				t_perplexity.start();
				double ppx_score = cal_perplexity_held_out();
				t_perplexity.stop();
				std::cout << std::fixed << std::setprecision(12) << "perplexity for hold out set is: " << ppx_score << std::endl;
				ppxs_held_out.push_back(ppx_score);
			}

			delete edgeSample.first;

			step_count++;
			t_outer.stop();
			auto l2 = std::chrono::system_clock::now();
			std::cout << "LOOP  = " << (l2-l1).count() << std::endl;
		}

		timer::Timer::printHeader(std::cout);
		std::cout << t_outer << std::endl;
		std::cout << t_perplexity << std::endl;
		std::cout << t_mini_batch << std::endl;
		std::cout << t_nodes_in_mini_batch << std::endl;
		std::cout << t_latent_vars << std::endl;
		std::cout << t_latent_vars2 << std::endl;
		std::cout << t_update_pi << std::endl;
		std::cout << t_update_beta << std::endl;
	}

protected:

#if defined CALCULATE_PHI_ON_DEVICE || defined RANDOM_FOLLOWS_SCALABLE_GRAPH
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

	void update_beta(const OrderedEdgeSet &mini_batch, double scale) {
		int arg;

		arg = 0;
		update_beta_calculate_theta_sum_kernel.setArg(arg++, clTheta);
		update_beta_calculate_theta_sum_kernel.setArg(arg++, clThetaSum);

		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_theta_sum_kernel,
											 cl::NullRange, cl::NDRange(kRoundedThreads), cl::NDRange(groupSize));
		clContext.queue.finish();

		::size_t countPartialSums = std::min(mini_batch.size(), globalThreads);
		::size_t calcGradsThreads = round_up_to_multiples(countPartialSums, groupSize);

		update_beta_calculate_grads_kernel.setArg(0, clBuffers);
		update_beta_calculate_grads_kernel.setArg(1, (cl_int)mini_batch.size());
		update_beta_calculate_grads_kernel.setArg(2, (cl_double)scale);
		update_beta_calculate_grads_kernel.setArg(3, (cl_int)countPartialSums);

		clContext.queue.finish(); // Wait for sample_latent_vars2

		cl::Event e_grads_kernel;
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_grads_kernel, cl::NullRange,
				cl::NDRange(calcGradsThreads), cl::NDRange(groupSize),
				NULL, &e_grads_kernel);

		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		cl_double2 clEta;
		clEta.s[0] = this->eta[0];
		clEta.s[1] = this->eta[1];

		e_grads_kernel.wait();

		update_beta_calculate_theta_kernel.setArg(0, clBuffers);
		update_beta_calculate_theta_kernel.setArg(1, (cl_double)scale);
		update_beta_calculate_theta_kernel.setArg(2, (cl_double)eps_t);
		update_beta_calculate_theta_kernel.setArg(3, clEta);
		update_beta_calculate_theta_kernel.setArg(4, (int)countPartialSums);

		clContext.queue.enqueueTask(update_beta_calculate_theta_kernel);
		clContext.queue.finish();

		arg = 0;
		update_beta_calculate_beta_kernel.setArg(arg++, clTheta);
		update_beta_calculate_beta_kernel.setArg(arg++, clBeta);

		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_beta_kernel, cl::NullRange, cl::NDRange(kRoundedThreads), cl::NDRange(groupSize));
		clContext.queue.finish();

#ifndef CALCULATE_PERPLEXITY_ON_DEVICE
		clContext.queue.enqueueReadBuffer(clBeta, CL_FALSE, 0, K * sizeof(double), beta.data());
#endif
		if (true) {
			clContext.queue.enqueueReadBuffer(clBeta, CL_FALSE, 0, K * sizeof(double), beta.data());
			std::cerr << __func__ << std::endl;
			std::cerr << "beta ";
			for (::size_t k = 0; k < K; k++) {
				std::cerr << beta[k] << " ";
			}
			std::cerr << std::endl;
		}
	}

	void sample_latent_vars2(const OrderedEdgeSet &mini_batch, const OrderedVertexSet &nodes) {
		::size_t i;
		std::vector<cl_int2> edges(mini_batch.size());
		// Copy edges
#ifndef NDEBUG
		std::vector<int> node_index(N, -1);
#else
		std::vector<int> node_index(N);
#endif
		i = 0;
		for (auto n: nodes) {
			node_index[n] = i;
			i++;
		}
#if 0
		std::transform(mini_batch.begin(), mini_batch.end(), edges.begin(), [](const Edge& e) {
			cl_int2 E;
			E.s[0] = node_index[e.first];
			E.s[1] = node_index[e.second];
			return E;
		});
#else
		i = 0;
		for (auto e: mini_batch) {
			assert(node_index[e.first] != -1);
			assert(node_index[e.second] != -1);
			edges[i].s[0] = node_index[e.first];
			edges[i].s[1] = node_index[e.second];
			i++;
		}
#endif
		clContext.queue.enqueueWriteBuffer(clEdges, CL_FALSE,
				0, edges.size()*sizeof(cl_int2),
				edges.data());

		::size_t arg = 0;
		sample_latent_vars2_kernel.setArg(arg++, clBuffers);
		sample_latent_vars2_kernel.setArg(arg++, (cl_int)edges.size());
		sample_latent_vars2_kernel.setArg(arg++, (cl_int)nodes.size());

		clContext.queue.finish(); // Wait for clEdges and PiUpdates from sample_latent_vars_and_update_pi

		clContext.queue.enqueueNDRangeKernel(sample_latent_vars2_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}

	void stage_subgraph(GraphWrapper *graph, const OrderedVertexSet &nodes) {
		// Copy held_out_set subgraph for nodes
		// ::size_t edge_elts = nodes.size() * std::max(network.get_max_fan_out(), num_node_sample + 1);
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
			if (node == 1218 || node == 918) {
				std::cerr << "Insert node " << node << " neighbors [";
				for (::size_t i = offset - neighbors.size(); i < offset; i++) {
					std::cerr << graph->h_subEdges[i] << " ";
				}
				std::cerr << "] offset becomes " << offset << std::endl;
			}
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
						   const OrderedVertexSet &nodes,
						   const std::vector<int> &neighbors) {
		hostBuffer.clear();
		for (auto n: nodes) {
			assert(data[n].size() == K);
			hostBuffer.insert(hostBuffer.end(), data[n].begin(), data[n].end());
		}
		for (auto n: neighbors) {
			assert(data[n].size() == K);
			hostBuffer.insert(hostBuffer.end(), data[n].begin(), data[n].end());
		}

		std::cerr << "Nodes[" << nodes.size() << "] Neighbors[" << neighbors.size() << "] host buffer[" << hostBuffer.size() << "]" << std::endl;
		clContext.queue.enqueueWriteBuffer(buffer, CL_FALSE, 0, (nodes.size() + neighbors.size()) * K * sizeof(cl_double), hostBuffer.data());
	}

	void retrieve_sub_vectors(cl::Buffer &buffer,
							  std::vector<std::vector<double> > &data,
							  std::vector<double> &hostBuffer,
							  const OrderedVertexSet &nodes) {

		clContext.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, nodes.size() * K * sizeof(cl_double), hostBuffer.data());

		::size_t i = 0;
		for (auto n: nodes) {
			data[n].clear();
			data[n].insert(data[n].end(), hostBuffer.begin() + i * K, hostBuffer.begin() + (i + 1) * K);
			std::cerr << "Update pi/phi[" << n << "]: ";
			for (auto x: data[n]) {
				std::cerr << std::fixed << std::setprecision(12) << x << " ";
			}
			std::cerr << std::endl;
			i++;
		}
	}

	void sample_latent_vars_neighbors(const OrderedVertexSet& nodes) {
		std::cerr << "Stage clSubHeldOutGraph" << std::endl;
		stage_subgraph(&clSubHeldOutGraph, nodes);

		// Copy sampled node IDs
		std::vector<cl_int> node_index(nodes.begin(), nodes.end()); // FIXME: replace OrderedVertexSet with vector
		clContext.queue.enqueueWriteBuffer(clNodes, CL_FALSE, 0, node_index.size() * sizeof(cl_int), node_index.data());

		int Idx = 0;
		sample_latent_vars_neighbors_kernel.setArg(Idx++, clBuffers);
		sample_latent_vars_neighbors_kernel.setArg(Idx++, (cl_int)nodes.size());

		clContext.queue.finish();
		clContext.queue.enqueueNDRangeKernel(sample_latent_vars_neighbors_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();

		clContext.queue.enqueueReadBuffer(clNodesNeighbors, CL_FALSE, 0, nodes.size() * real_num_node_sample() * sizeof(cl_int), hostNeighbors.data());
	}

	void sample_latent_vars_sample_z_ab(const OrderedVertexSet& nodes) {
		std::cerr << "Stage clSubGraph" << std::endl;
		stage_subgraph(&clSubGraph, nodes);

		// TODO FIXME TODO FIXME
		// On the host, just maintain Phi.
		// Stage only the sub_vectors for Phi. Calculate Pi from these.
		// No need to maintain Pi on the host, or to copy it back and forth.
		// TODO FIXME TODO FIXME
		//
		// Pi is staged in two contiguous sections:
		// 1) pi(i) for i in nodes
		// 2) pi(n(i)) for the neighbors n(i) of node i
		stage_sub_vectors(clPi, pi, hostPiBuffer, nodes, hostNeighbors);

		int Idx = 0;
		sample_latent_vars_sample_z_ab_kernel.setArg(Idx++, clBuffers);
		sample_latent_vars_sample_z_ab_kernel.setArg(Idx++, (cl_int)nodes.size());
		sample_latent_vars_sample_z_ab_kernel.setArg(Idx++, (cl_double)epsilon);

		clContext.queue.finish();
		check_for_kernel_errors(); // sample_latent_vars
		clContext.queue.enqueueNDRangeKernel(sample_latent_vars_sample_z_ab_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
		check_for_kernel_errors(); // sample_latent_vars
	}

	void sample_latent_vars(const OrderedVertexSet& nodes) {
		sample_latent_vars_neighbors(nodes);
		sample_latent_vars_sample_z_ab(nodes);
	}

	void update_pi(const OrderedVertexSet& nodes) {

		// 1. merge nodes and neighbors
		// 2. stage pi for the merged(nodes, neighbors)
		// 3. ensure that pi is staged in the order of the device neighbor index array
		stage_sub_vectors(clPhi, phi, hostPhiBuffer, nodes, hostNeighbors);

		int Idx = 0;
		update_pi_kernel.setArg(Idx++, clBuffers);
		update_pi_kernel.setArg(Idx++, (cl_int)nodes.size());
		update_pi_kernel.setArg(Idx++, (cl_double)alpha);
		update_pi_kernel.setArg(Idx++, (cl_double)a);
		update_pi_kernel.setArg(Idx++, (cl_double)b);
		update_pi_kernel.setArg(Idx++, (cl_double)c);
		update_pi_kernel.setArg(Idx++, (cl_int)step_count);
		update_pi_kernel.setArg(Idx++, (cl_int)N);

		clContext.queue.finish();
		check_for_kernel_errors(); // sample_latent_vars
		clContext.queue.enqueueNDRangeKernel(update_pi_kernel, cl::NullRange, cl::NDRange(globalThreads), cl::NDRange(groupSize));
		clContext.queue.finish();
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
		retrieve_sub_vectors(clPi, pi, hostPiBuffer, nodes);
		retrieve_sub_vectors(clPhi, phi, hostPhiBuffer, nodes);
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

#ifdef CALCULATE_PERPLEXITY_ON_DEVICE
	double cal_perplexity_held_out() {
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
#endif


	// TODO: Deprecate
	// FIXME: Why is linkedMap not a ref?
	void _prepare_flat_cl_graph(cl::Buffer &edges, cl::Buffer &nodes, cl::Buffer &graph, std::map<int, std::vector<int>> linkedMap) {
		const int h_edges_size = 2 * network.get_num_linked_edges();
		std::unique_ptr<cl_int[]> h_edges(new cl_int[h_edges_size]);
		const int h_nodes_size = network.get_num_nodes();
		std::unique_ptr<cl_int2[]> h_nodes(new cl_int2[h_nodes_size]);

		size_t offset = 0;
		for (cl_int i = 0; i < network.get_num_nodes(); ++i) {
			auto it = linkedMap.find(i);
			if (it == linkedMap.end()) {
				h_nodes.get()[i].s[0] = 0;
				h_nodes.get()[i].s[1] = 0;
			} else {
				std::sort(it->second.begin(), it->second.end());
				h_nodes.get()[i].s[0] = it->second.size();
				h_nodes.get()[i].s[1] = offset;
				for (auto viter = it->second.begin();
						viter != it->second.end(); ++viter) {
					h_edges.get()[offset] = *viter;
					++offset;
				}
			}
		}

		edges = createBuffer("graph edges", CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_edges_size*sizeof(cl_int), h_edges.get());
		std::cerr << "create edge buffer size " << h_edges_size << " entries" << std::endl;
		nodes = createBuffer("graph nodes", CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_nodes_size*sizeof(cl_int2), h_nodes.get());
		std::cerr << "create node buffer size " << h_nodes_size << " entries" << std::endl;
		graph = createBuffer("graph", CL_MEM_READ_WRITE, 2*64/8 /* 2 pointers, each is at most 64-bits */);

		graph_init_kernel.setArg(0, graph);
		graph_init_kernel.setArg(1, edges);
		graph_init_kernel.setArg(2, nodes);

		clContext.queue.enqueueTask(graph_init_kernel);
		clContext.queue.finish();
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
		graph_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");
		std::map<int, std::vector<int>> linkedMap;

		for (auto e : network.get_linked_edges()) {
			linkedMap[e.first].push_back(e.second);
			linkedMap[e.second].push_back(e.first);
		}
		_prepare_flat_cl_graph(clGraphEdges, clGraphNodes, clGraph, linkedMap);
		std::cerr << "Get adjacency list repr of Graph" << std::endl;

		// BASED ON STRATIFIED RANDOM NODE SAMPLING STRATEGY
		::size_t num_nodes_in_batch = N/10 + 1 + 1;
		::size_t num_edges_in_batch = std::min(num_nodes_in_batch * network.get_max_fan_out(), network.get_num_linked_edges());
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
		_prepare_flat_cl_graph(clHeldOutGraphEdges, clHeldOutGraphNodes, clHeldOutGraph, linkedMap);
		std::cerr << "Get adjacency list repr of HeldOutGraph" << std::endl;
		// BASED ON STRATIFIED RANDOM NODE SAMPLING STRATEGY
		num_edges_in_batch = num_nodes_in_batch - 1;
		clSubHeldOutGraph.init(*this, &clHeldOutGraph, linkedMap, N, num_nodes_in_batch, num_edges_in_batch);

		prepare_iterable_cl_graph(clIterableHeldOutGraph, network.get_held_out_set());

#if MCMC_CL_STOCHASTIC_TEST_GRAPH
		test_graph();
#endif
	}

#if MCMC_CL_STOCHASTIC_TEST_GRAPH // TEST GRAPH ON OPENCL's SIDE
	void test_graph() {
		cl::Program program2 = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/test.cl", progOpts);
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
#ifndef CALCULATE_PHI_ON_DEVICE
	std::vector<std::vector<double> > phi;			// parameterization for \pi
#endif

	std::string progOpts;

	cl::ClContext clContext;

	cl::Program graph_program;
	cl::Program sampler_program;

	cl::Kernel graph_init_kernel;
	cl::Kernel random_gamma_kernel;
	cl::Kernel row_normalize_kernel;
	cl::Kernel sample_latent_vars_neighbors_kernel;
	cl::Kernel sample_latent_vars_sample_z_ab_kernel;
	cl::Kernel update_pi_kernel;
	cl::Kernel sample_latent_vars2_kernel;
	cl::Kernel update_beta_calculate_theta_sum_kernel;
	cl::Kernel update_beta_calculate_grads_kernel;
	cl::Kernel update_beta_calculate_theta_kernel;
	cl::Kernel update_beta_calculate_beta_kernel;
#ifdef CALCULATE_PERPLEXITY_ON_DEVICE
	cl::Kernel cal_perplexity_kernel;
#endif
	cl::Kernel init_buffers_kernel;

	cl::Buffer clBuffers;

	GraphWrapper clSubGraph;
	GraphWrapper clSubHeldOutGraph;

	cl::Buffer clIterableHeldOutGraph;

	cl::Buffer clGraphEdges;				// to be deprecated
	cl::Buffer clGraphNodes;				// to be deprecated
	cl::Buffer clGraph;				// to be deprecated
	cl::Buffer clHeldOutGraphEdges;		// to be deprecated
	cl::Buffer clHeldOutGraphNodes;		// to be deprecated
	cl::Buffer clHeldOutGraph;		// to be deprecated
	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clNodesNeighborsHash;
	cl::Buffer clEdges;
	cl::Buffer clPi;
	cl::Buffer clPhi;
	cl::Buffer clBeta;
	cl::Buffer clTheta;
	cl::Buffer clThetaSum;
	cl::Buffer clZ;
	cl::Buffer clScratch;
	cl::Buffer clRandomSeed;
	cl::Buffer clErrorCtrl;
	cl::Buffer clErrorMsg;
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
	cl::Buffer clStoredRandom;
#endif

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

	// To be deprecated:
	// std::vector<std::vector<double> > theta;		// parameterization for \beta
	// To be deprecated:
	// std::vector<std::vector<double> > phi;			// parameterization for \pi
	Random::Random kernelRandom;

	const ::size_t groupSize;
	const ::size_t numGroups;
	const ::size_t globalThreads;
	const ::size_t kRoundedThreads;

	::size_t totalAllocedClBufers;
	std::vector<std::pair<std::string, ::size_t> > clBufAllocSizes;

	char *errMsg;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
