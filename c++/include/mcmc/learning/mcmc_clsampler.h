#ifndef MCMC_LEARNING_MCMC_CLSAMPLER_H_
#define MCMC_LEARNING_MCMC_CLSAMPLER_H_


#include "opencl/context.h"

#include "mcmc/learning/mcmc_sampler.h"


namespace mcmc {
namespace learning {

#define stringify(str)	#str

class MCMCClSampler : virtual public MCMCSampler {
public:
	MCMCClSampler(const Options &args, const Network &network, const cl::ClContext clContext)
		: Learner(args, network), MCMCSampler(args, network), clContext(clContext) {

		std::ostringstream opts;
		opts << "-IOpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << real_num_node_sample()
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cout << "num_node_sample = " << num_node_sample << std::endl;

		init_graph();

		clNodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * sizeof(cl_int) // max: 2 unique nodes per edge in batch
				// FIXME: better estimate for #nodes in mini batch
				);
		clNodesNeighbors = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * real_num_node_sample() * sizeof(cl_int) // #total_nodes x #neighbors_per_node
				// FIXME: we need space for all N elements. Space should be limited to #nodes_in_mini_batch * num_node_sample (DEPENDS ON ABOVE)
				);
		clPi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clPhi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clBeta = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #total_nodes x #K
				);
		clZ = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_int) // #total_nodes x #K
				);
		clRandom = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * std::max(K, real_num_node_sample()) * sizeof(cl_double) // at most #total_nodes
				);
		clScratch = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
	}

	virtual ~MCMCClSampler() {
	}

protected:
	void init_graph() {
		const int h_edges_size = 2 * network.get_num_linked_edges();
		std::unique_ptr<cl_int[]> h_edges(new cl_int[h_edges_size]);
		const int h_nodes_size = network.get_num_nodes();
		std::unique_ptr<cl_int2[]> h_nodes(new cl_int2[h_nodes_size]);

		std::map<int, std::vector<int>> linkedMap;
		for (auto e : network.get_linked_edges()) {
			linkedMap[e.first].push_back(e.second);
			linkedMap[e.second].push_back(e.first);
		}

		size_t offset = 0;
		for (cl_int i = 0; i < network.get_num_nodes(); ++i) {
			auto it = linkedMap.find(i);
			if (it == linkedMap.end()) {
				h_nodes.get()[i].s[0] = 0;
				h_nodes.get()[i].s[1] = 0;
			} else {
				h_nodes.get()[i].s[0] = it->second.size();
				h_nodes.get()[i].s[1] = offset;
				for (auto viter = it->second.begin();
						viter != it->second.end(); ++viter) {
					h_edges.get()[offset] = *viter;
					++offset;
				}
			}
		}

		clGraphEdges = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_edges_size*sizeof(cl_int), h_edges.get());
		clGraphNodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_nodes_size*sizeof(cl_int2), h_nodes.get());
		clGraph = cl::Buffer(clContext.context, CL_MEM_READ_WRITE, 2*64/8 /* 2 pointers, each is at most 64-bits */);

		graph_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/OpenCL/graph.cl", progOpts);
		graph_init_kernel = cl::Kernel(graph_program, "graph_init");
		graph_init_kernel.setArg(0, clGraph);
		graph_init_kernel.setArg(1, clGraphEdges);
		graph_init_kernel.setArg(2, clGraphNodes);

		clContext.queue.enqueueTask(graph_init_kernel);
		clContext.queue.finish();
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

	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clPi;
	cl::Buffer clPhi;
	cl::Buffer clBeta;
	cl::Buffer clZ;
	cl::Buffer clRandom;
	cl::Buffer clScratch;
};

} // namespace learning
} // namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_CLSAMPLER_H_
