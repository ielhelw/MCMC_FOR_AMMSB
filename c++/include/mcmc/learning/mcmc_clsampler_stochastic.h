#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include "mcmc_sampler_stochastic.h"

#include "opencl/context.h"


namespace mcmc {
namespace learning {

class MCMCClSamplerStochastic : public MCMCSamplerStochastic {
public:
	MCMCClSamplerStochastic(const Options &args, const Network &graph, const cl::ClContext clContext)
		: MCMCSamplerStochastic(args, graph), clContext(clContext) {

		const int h_edges_size = 2 * graph.get_num_linked_edges();
		cl_int* h_edges = new cl_int[h_edges_size];
		const int h_nodes_size = graph.get_num_nodes();
		cl_int2* h_nodes = new cl_int2[h_nodes_size];

		std::map<int, std::vector<int>> linkedMap;
		for (auto e : graph.get_linked_edges()) {
			linkedMap[e.first].push_back(e.second);
			linkedMap[e.second].push_back(e.first);
		}

		size_t offset = 0;
		for (cl_int i = 0; i < graph.get_num_nodes(); ++i) {
			auto it = linkedMap.find(i);
			if (it == linkedMap.end()) {
				h_nodes[i].s[0] = 0;
				h_nodes[i].s[1] = 0;
			} else {
				h_nodes[i].s[0] = it->second.size();
				h_nodes[i].s[1] = offset;
				for (auto viter = it->second.begin();
						viter != it->second.end(); ++viter) {
					h_edges[offset] = *viter;
					++offset;
				}
			}
		}

		cl::Buffer clGraphEdges = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_edges_size*sizeof(cl_int), h_edges);
		cl::Buffer clGraphNodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, h_nodes_size*sizeof(cl_int2), h_nodes);
		cl::Buffer clGraph = cl::Buffer(clContext.context, CL_MEM_READ_WRITE, 2*64/8 /* 2 pointers, each is at most 64-bits */);

		std::ostringstream opts;
		opts << "-IOpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << args.mini_batch_size;
		std::string progOpts = opts.str();

		cl::Program program = this->clContext.createProgram("OpenCL/graph.cl", progOpts);
		cl::Kernel graph_init_kernel(program, "graph_init");
		graph_init_kernel.setArg(0, clGraph);
		graph_init_kernel.setArg(1, clGraphEdges);
		graph_init_kernel.setArg(2, clGraphNodes);

		clContext.queue.enqueueTask(graph_init_kernel);
		clContext.queue.finish();

#if 0 // TEST GRAPH ON OPENCL's SIDE
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
#endif
	}

	virtual void run() {
	        /** run mini-batch based MCMC sampler, based on the sungjin's note */
	        while (step_count < max_iteration && ! is_converged()) {
				EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
				const EdgeSet &mini_batch = *edgeSample.first;
				double scale = edgeSample.second;

				std::unordered_map<int, std::vector<double> > latent_vars;
				std::unordered_map<int, ::size_t> size;
				std::unordered_map<int, OrderedVertexSet> neighbor_nodes;

	            // iterate through each node in the mini batch.
				OrderedVertexSet nodes = nodes_in_batch(mini_batch);

				// pre-generate neighbors for each node
	            for (auto node = nodes.begin();
					 	node != nodes.end();
						node++) {
	                // sample a mini-batch of neighbors
	            	neighbor_nodes[*node] = sample_neighbor_nodes(num_node_sample, *node);
	                size[*node] = neighbor_nodes[*node].size();
	            }
	            // execute kernels for all
	            for (auto node = nodes.begin();
						node != nodes.end();
						node++){
	                // sample latent variables z_ab for each pair of nodes
	                std::vector<double> z = this->sample_latent_vars(*node, neighbor_nodes[*node], /* FIXME */ false);
	                // save for a while, in order to update together.
	                latent_vars[*node] = z;
				}

	            // update pi for each node
	            for (auto node = nodes.begin();
					 	node != nodes.end();
						node++) {
	                update_pi_for_node(*node, latent_vars[*node], size[*node], scale);
				}

	            // sample (z_ab, z_ba) for each edge in the mini_batch.
	            // z is map structure. i.e  z = {(1,10):3, (2,4):-1}
				EdgeMapZ z = sample_latent_vars2(mini_batch);
	            update_beta(mini_batch, scale, z);


	            if (step_count % 1 == 0) {
	                double ppx_score = cal_perplexity_held_out();
					std::cout << "perplexity for hold out set is: " << ppx_score << std::endl;
	                ppxs_held_out.push_back(ppx_score);
				}

				std::cerr << "GC mini_batch->first EdgeSet *" << std::endl;
				delete edgeSample.first;

	            step_count++;
			}
		}


protected:
	cl::ClContext clContext;
	cl::Kernel sample_latent_vars_kernel;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
