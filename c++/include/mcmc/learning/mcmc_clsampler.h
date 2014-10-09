#ifndef MCMC_LEARNING_MCMC_CLSAMPLER_H_
#define MCMC_LEARNING_MCMC_CLSAMPLER_H_


#include "opencl/context.h"

#include "mcmc/learning/mcmc_sampler.h"

#define PARALLELISM	1


namespace mcmc {
namespace learning {

#define do_stringify(str)	#str
#define stringify(str)		do_stringify(str)

class MCMCClSampler : public MCMCSampler {
public:
	MCMCClSampler(const Options &args, const Network &network,
				  ::size_t num_node_sampler, double eta0, double eta1,
				  const cl::ClContext clContext)
		: MCMCSampler(args, network, num_node_sampler, eta0, eta1), clContext(clContext) {

		int hash_table_multiple = 2;

		std::ostringstream opts;
		opts << "-I" << stringify(PROJECT_HOME) << "/../OpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << real_num_node_sample()
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N
			 << " -DRAND_MAX=" << std::numeric_limits<uint64_t>::max()
#ifdef RANDOM_FOLLOWS_CPP
			 << " -DRANDOM_FOLLOWS_CPP"
#endif
			 << " -DHASH_MULTIPLE=" << hash_table_multiple;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cout << "num_node_sample = " << num_node_sample << std::endl;

		init_graph();

		clNodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * sizeof(cl_int) // max: 2 unique nodes per edge in batch
				// FIXME: better estimate for #nodes in mini batch
				);
		clNodesNeighbors = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * real_num_node_sample() * hash_table_multiple * sizeof(cl_int) // #total_nodes x #neighbors_per_node
				// FIXME: we need space for all N elements. Space should be limited to #nodes_in_mini_batch * num_node_sample (DEPENDS ON ABOVE)
				);
		clEdges = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * sizeof(cl_int2) // #total_nodes x #neighbors_per_node
				// FIXME: should be multiple of mini_batch_size
				);
		clPi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clPiUpdate = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clPhi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clBeta = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #total_nodes x #K
				);
		clTheta = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				2 * K * sizeof(cl_double) // 2 x #K
				);
		clThetaSum = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #K
				);
		clZ = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_int) // #total_nodes x #K
				);
		clScratch = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				std::max(
						N * K * sizeof(cl_double), // #total_nodes x #K
						N * K * sizeof(cl_double3)
					)
				);
		clRandomSeed = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				PARALLELISM * sizeof(cl_ulong2)
				);
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

		std::vector<cl_ulong2> randomSeed(PARALLELISM);
		for (unsigned int i = 0; i < randomSeed.size(); ++i) {
			randomSeed[i].s[0] = 42 + i;
			randomSeed[i].s[1] = 42 + i + 1;
		}
		clContext.queue.enqueueWriteBuffer(clRandomSeed, CL_TRUE,
				0, randomSeed.size() * sizeof(cl_ulong2),
				randomSeed.data());
		clContext.queue.enqueueFillBuffer(clNodesNeighbors, (cl_int)-1, 0, clNodesNeighbors.getInfo<CL_MEM_SIZE>());
		clContext.queue.finish();
	}

	virtual ~MCMCClSampler() {
	}

protected:
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

		edges = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_edges_size*sizeof(cl_int), h_edges.get());
		nodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_nodes_size*sizeof(cl_int2), h_nodes.get());
		graph = cl::Buffer(clContext.context, CL_MEM_READ_WRITE, 2*64/8 /* 2 pointers, each is at most 64-bits */);

		graph_init_kernel.setArg(0, graph);
		graph_init_kernel.setArg(1, edges);
		graph_init_kernel.setArg(2, nodes);

		clContext.queue.enqueueTask(graph_init_kernel);
		clContext.queue.finish();
	}

	void init_graph() {
		graph_program = this->clContext.createProgram("OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");
		std::map<int, std::vector<int>> linkedMap;

		for (auto e : network.get_linked_edges()) {
			linkedMap[e.first].push_back(e.second);
			linkedMap[e.second].push_back(e.first);
		}
		_prepare_flat_cl_graph(clGraphEdges, clGraphNodes, clGraph, linkedMap);

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


#if MCMC_CL_STOCHASTIC_TEST_GRAPH
		test_graph();
#endif
	}

#if MCMC_CL_STOCHASTIC_TEST_GRAPH // TEST GRAPH ON OPENCL's SIDE
	void test_graph() {
		cl::Program program2 = this->clContext.createProgram("OpenCL/test.cl", progOpts);
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


protected:
	std::string progOpts;

	cl::ClContext clContext;

	cl::Program graph_program;

	cl::Kernel graph_init_kernel;

	cl::Buffer clGraphEdges;
	cl::Buffer clGraphNodes;
	cl::Buffer clGraph;

	cl::Buffer clHeldOutGraphEdges;
	cl::Buffer clHeldOutGraphNodes;
	cl::Buffer clHeldOutGraph;

	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clEdges;
	cl::Buffer clPi;
	cl::Buffer clPiUpdate;
	cl::Buffer clPhi;
	cl::Buffer clBeta;
	cl::Buffer clTheta;
	cl::Buffer clThetaSum;
	cl::Buffer clZ;
	cl::Buffer clScratch;
	cl::Buffer clRandomSeed;
};

} // namespace learning
} // namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_CLSAMPLER_H_
