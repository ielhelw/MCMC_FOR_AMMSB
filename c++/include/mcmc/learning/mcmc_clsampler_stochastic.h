#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>

#include "mcmc_sampler_stochastic.h"

#include "opencl/context.h"


namespace mcmc {
namespace learning {

class MCMCClSamplerStochastic : public MCMCSamplerStochastic {
public:
	MCMCClSamplerStochastic(const Options &args, const Network &graph, const cl::ClContext clContext)
		: MCMCSamplerStochastic(args, graph), clContext(clContext) {

		std::ostringstream opts;
		opts << "-IOpenCL/include"
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << num_node_sample
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cout << "num_node_sample = " << num_node_sample << std::endl;

		init_graph();

		sampler_program = this->clContext.createProgram("OpenCL/sampler.cl", progOpts);
		sample_latent_vars_kernel = cl::Kernel(sampler_program, "sample_latent_vars");

		clNodes = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * sizeof(cl_int) // max: 2 unique nodes per edge in batch
				// FIXME: better estimate for #nodes in mini batch
				);
		clNodesNeighbors = cl::Buffer(clContext.context, CL_MEM_READ_ONLY,
				N * num_node_sample * sizeof(cl_int) // #total_nodes x #neighbors_per_node
				// FIXME: we need space for all N elements. Space should be limited to #nodes_in_mini_batch * num_node_sample (DEPENDS ON ABOVE)
				);
		clPi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(double) // #total_nodes x #K
				);
		clBeta = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				K * sizeof(double) // #total_nodes x #K
				);
		clZ = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(double) // #total_nodes x #K
				);
		clRandom = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * sizeof(double) // at most #total_nodes
				);
		clScratchP = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(double) // #total_nodes x #K
				);
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

	            // Copy sampled node IDs
	            std::vector<int> v_nodes(nodes.begin(), nodes.end()); // FIXME: replace OrderedVertexSet with vector
            	clContext.queue.enqueueWriteBuffer(clNodes, CL_TRUE, 0, v_nodes.size()*sizeof(int), &(v_nodes[0]));

	            // Copy neighbors of *sampled* nodes only
	            for (auto node = nodes.begin(); node != nodes.end(); ++node) {
	            	std::vector<int> neighbors(neighbor_nodes[*node].begin(), neighbor_nodes[*node].end());
	            	clContext.queue.enqueueWriteBuffer(clNodesNeighbors, CL_TRUE,
	            			*node * num_node_sample * sizeof(cl_int),
	            			num_node_sample * sizeof(cl_int),
	            			&(neighbors[0]));
	            }

	            // Copy pi
	            for (unsigned int i = 0; i < pi.size(); ++i) {
	            	clContext.queue.enqueueWriteBuffer(clPi, CL_TRUE,
	            			i * K * sizeof(double),
	            			K * sizeof(double),
	            			&(pi[i][0]));
	            }

	            // Copy beta
	            clContext.queue.enqueueWriteBuffer(clBeta, CL_TRUE, 0, K * sizeof(double), &beta[0]);

	            // Generate and copy Randoms
	            std::vector<double> randoms(nodes.size());
	            std::generate(randoms.begin(), randoms.end(), std::bind(&Random::FileReaderRandom::random, Random::random));
	            clContext.queue.enqueueWriteBuffer(clRandom, CL_TRUE, 0, nodes.size()*sizeof(cl_double), &(randoms[0]));

	            sample_latent_vars_kernel.setArg(0, clGraph);
	            sample_latent_vars_kernel.setArg(1, clNodes);
	            sample_latent_vars_kernel.setArg(2, (cl_int)nodes.size());
	            sample_latent_vars_kernel.setArg(3, clNodesNeighbors);
	            sample_latent_vars_kernel.setArg(4, clPi);
	            sample_latent_vars_kernel.setArg(5, clBeta);
	            sample_latent_vars_kernel.setArg(6, (cl_double)epsilon);
	            sample_latent_vars_kernel.setArg(7, clZ);
	            sample_latent_vars_kernel.setArg(8, clRandom);
	            sample_latent_vars_kernel.setArg(9, clScratchP);

	            // FIXME: threading granularity
	            clContext.queue.enqueueNDRangeKernel(sample_latent_vars_kernel, cl::NullRange, cl::NDRange(4), cl::NDRange(1));
	            clContext.queue.finish();

	            std::unordered_map<int, std::vector<double>> zMap;
	            for (auto node = nodes.begin(); node != nodes.end(); ++node) {
	            	zMap[*node] = std::vector<double>(K, 0);
	            	clContext.queue.enqueueReadBuffer(clZ, CL_TRUE,
	            			(*node) * K * sizeof(double),
	            			K * sizeof(double),
	            			&(zMap[*node][0]));
	            	latent_vars[*node] = zMap[*node];
	            }

#if 0
	            // execute kernels for all
	            for (auto node = nodes.begin();
						node != nodes.end();
						node++){
	                // sample latent variables z_ab for each pair of nodes
	                std::vector<double> z = this->sample_latent_vars(*node, neighbor_nodes[*node], /* FIXME */ false);
	                if (!std::equal(z.begin(), z.end(), zMap[*node].begin())) {
	                	abort();
	                }
	                // save for a while, in order to update together.
	                latent_vars[*node] = z;
				}
#endif

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

		graph_program = this->clContext.createProgram("OpenCL/graph.cl", progOpts);
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

	std::string progOpts;

	cl::ClContext clContext;

	cl::Program graph_program;
	cl::Program sampler_program;

	cl::Kernel graph_init_kernel;
	cl::Kernel sample_latent_vars_kernel;

	cl::Buffer clGraphEdges;
	cl::Buffer clGraphNodes;
	cl::Buffer clGraph;

	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clPi;
	cl::Buffer clBeta;
	cl::Buffer clZ;
	cl::Buffer clRandom;
	cl::Buffer clScratchP;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
