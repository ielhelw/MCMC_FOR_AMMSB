#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>
#include <chrono>

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
			 << " -DNEIGHBOR_SAMPLE_SIZE=" << real_num_node_sample()
			 << " -DK=" << K
			 << " -DMAX_NODE_ID=" << N;
		progOpts = opts.str();

		std::cout << "COMPILE OPTS: " << progOpts << std::endl;

		std::cout << "num_node_sample = " << num_node_sample << std::endl;

		init_graph();

		sampler_program = this->clContext.createProgram("OpenCL/sampler.cl", progOpts);
		sample_latent_vars_and_update_pi_kernel = cl::Kernel(sampler_program, "sample_latent_vars_and_update_pi");
		sample_latent_vars2_kernel = cl::Kernel(sampler_program, "sample_latent_vars2");
		update_beta_calculate_grads_kernel = cl::Kernel(sampler_program, "update_beta_calculate_grads");
		update_beta_calculate_theta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta");

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
		clPiUpdate = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clPhi = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // #total_nodes x #K
				);
		clBeta = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				K * sizeof(cl_double) // #K
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
		clRandomNK = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * K * sizeof(cl_double) // at most #total_nodes
				);
		clRandomNN = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				N * real_num_node_sample() * sizeof(cl_double) // at most #total_nodes
				);
		clScratch = cl::Buffer(clContext.context, CL_MEM_READ_WRITE,
				std::max(
						N * K * sizeof(cl_double), // #total_nodes x #K
						N * K * sizeof(cl_double3)
					)
				);
		for (unsigned i = 0; i < N; ++i) {
			clContext.queue.enqueueWriteBuffer(clPhi, CL_TRUE,
					i * K * sizeof(cl_double),
					K * sizeof(cl_double),
					phi[i].data());
		}

		info(std::cout);
	}

protected:

	virtual void update_beta(const OrderedEdgeSet &mini_batch, double scale, const EdgeMapZ &z) {

		// copy edges: already done in sample_latent_vars2

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

		// copy theta_sum
		std::vector<cl_double> vThetaSum(theta.size());
		std::transform(theta.begin(), theta.end(), vThetaSum.begin(), np::sum<double>);
		clContext.queue.enqueueWriteBuffer(clThetaSum, CL_TRUE,
				0, theta.size() * sizeof(cl_double),
				vThetaSum.data());

		update_beta_calculate_grads_kernel.setArg(0, clGraph);
		update_beta_calculate_grads_kernel.setArg(1, clNodesNeighbors);
		update_beta_calculate_grads_kernel.setArg(2, (cl_int)mini_batch.size());
		update_beta_calculate_grads_kernel.setArg(3, clZ);
		update_beta_calculate_grads_kernel.setArg(4, clTheta);
		update_beta_calculate_grads_kernel.setArg(5, clThetaSum);
		update_beta_calculate_grads_kernel.setArg(6, clScratch);
		update_beta_calculate_grads_kernel.setArg(7, (cl_double)scale);

		::size_t countPartialSums = std::min(mini_batch.size(), (::size_t)2);

		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_grads_kernel, cl::NullRange, cl::NDRange(countPartialSums), cl::NDRange(1));
		clContext.queue.finish();

		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		cl_double2 clEta;
		clEta.s[0] = this->eta[0];
		clEta.s[1] = this->eta[1];

		std::vector<std::vector<double> > noise = Random::random->randn(K, 2);	// random noise.
		std::vector<cl_double2> clNoise(noise.size());
		std::transform(noise.begin(), noise.end(), clNoise.begin(), [](const std::vector<double>& n){
			cl_double2 ret;
			ret.s[0] = n[0];
			ret.s[1] = n[1];
			return ret;
		});

		clContext.queue.enqueueWriteBuffer(clRandomNK, CL_TRUE,
				0, clNoise.size()*sizeof(cl_double2),
				clNoise.data());

		update_beta_calculate_theta_kernel.setArg(0, clTheta);
		update_beta_calculate_theta_kernel.setArg(1, clRandomNK);
		update_beta_calculate_theta_kernel.setArg(2, clScratch);
		update_beta_calculate_theta_kernel.setArg(3, (cl_double)scale);
		update_beta_calculate_theta_kernel.setArg(4, (cl_double)eps_t);
		update_beta_calculate_theta_kernel.setArg(5, clEta);
		update_beta_calculate_theta_kernel.setArg(6, (int)countPartialSums);

		clContext.queue.enqueueTask(update_beta_calculate_theta_kernel);
		clContext.queue.finish();


		// copy theta
		clContext.queue.enqueueReadBuffer(clTheta, CL_TRUE,
				0, theta.size() * sizeof(cl_double2),
				vclTheta.data());
		std::transform(vclTheta.begin(), vclTheta.end(), theta.begin(), [](const cl_double2 in){
			std::vector<double> ret(2);
			ret[0] = in.s[0];
			ret[1] = in.s[1];
			return ret;
		});

		std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
		np::row_normalize(&temp, theta);
		std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
	}

	virtual EdgeMapZ sample_latent_vars2(const OrderedEdgeSet &mini_batch) {
		std::vector<cl_int2> edges(mini_batch.size());
		// Copy edges
		std::transform(mini_batch.begin(), mini_batch.end(), edges.begin(), [](const Edge& e) {
			cl_int2 E;
			E.s[0] = e.first;
			E.s[1] = e.second;
			return E;
		});
		clContext.queue.enqueueWriteBuffer(clNodesNeighbors, CL_TRUE,
				0, edges.size()*sizeof(cl_int2),
				edges.data());

		// Generate and copy Randoms
		std::vector<double> randoms(edges.size());
#ifdef RANDOM_FOLLOWS_PYTHON
		std::generate(randoms.begin(), randoms.end(), std::bind(&Random::FileReaderRandom::random, Random::random));
#else
		std::generate(randoms.begin(), randoms.end(), std::bind(&Random::Random::random, Random::random));
#endif
		clContext.queue.enqueueWriteBuffer(clRandomNN, CL_TRUE,
				0, edges.size() * sizeof(cl_double),
				randoms.data());

		sample_latent_vars2_kernel.setArg(0, clGraph);
		sample_latent_vars2_kernel.setArg(1, clNodesNeighbors);
		sample_latent_vars2_kernel.setArg(2, (cl_int)edges.size());
		sample_latent_vars2_kernel.setArg(3, clPi);
		sample_latent_vars2_kernel.setArg(4, clBeta);
		sample_latent_vars2_kernel.setArg(5, clZ);
		sample_latent_vars2_kernel.setArg(6, clScratch);
		sample_latent_vars2_kernel.setArg(7, clRandomNN);

		clContext.queue.enqueueNDRangeKernel(sample_latent_vars2_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
		clContext.queue.finish();

		EdgeMapZ ezm;
		// No need to copy Zs back to host space
		return ezm;
	}

	virtual void update_pi_for_node_stub(const OrderedVertexSet& __attribute__((unused)) nodes,
			std::unordered_map<int, ::size_t>& __attribute__((unused)) size,
			std::unordered_map<int, std::vector<int> >& __attribute__((unused)) latent_vars,
			double scale) {
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}

	virtual void sample_latent_vars_stub(const OrderedVertexSet& nodes,
			std::unordered_map<int, ::size_t>& size,
			std::unordered_map<int, std::vector<int> >& latent_vars) {

		std::unordered_map<int, OrderedVertexSet> neighbor_nodes;
		std::vector<double> randoms;
		// pre-generate neighbors for each node
		for (auto node = nodes.begin();
				node != nodes.end();
				node++) {
			// sample a mini-batch of neighbors
			neighbor_nodes[*node] = sample_neighbor_nodes(num_node_sample, *node);
			size[*node] = neighbor_nodes[*node].size();

			// Generate Randoms
			std::vector<double> rs(real_num_node_sample());
#ifdef RANDOM_FOLLOWS_PYTHON
			std::generate(rs.begin(), rs.end(), std::bind(&Random::FileReaderRandom::random, Random::random));
#else
			std::generate(rs.begin(), rs.end(), std::bind(&Random::Random::random, Random::random));
#endif
			randoms.insert(randoms.end(), rs.begin(), rs.end());
		}

		// Copy sampled node IDs
		std::vector<int> v_nodes(nodes.begin(), nodes.end()); // FIXME: replace OrderedVertexSet with vector
		clContext.queue.enqueueWriteBuffer(clNodes, CL_TRUE, 0, v_nodes.size()*sizeof(int), v_nodes.data());

		// Copy neighbors of *sampled* nodes only
		for (auto node = nodes.begin(); node != nodes.end(); ++node) {
			std::vector<int> neighbors(neighbor_nodes[*node].begin(), neighbor_nodes[*node].end()); // FIXME: replace OrderedVertexSet with vector
			clContext.queue.enqueueWriteBuffer(clNodesNeighbors, CL_TRUE,
					*node * real_num_node_sample() * sizeof(cl_int),
					real_num_node_sample() * sizeof(cl_int),
					neighbors.data());
		}

		// Copy pi
		for (unsigned int i = 0; i < pi.size(); ++i) {
			clContext.queue.enqueueWriteBuffer(clPi, CL_TRUE,
					i * K * sizeof(double),
					K * sizeof(double),
					pi[i].data());
		}

		// Copy beta
		clContext.queue.enqueueWriteBuffer(clBeta, CL_TRUE, 0, K * sizeof(double), beta.data());

		// Copy Randoms
		clContext.queue.enqueueWriteBuffer(clRandomNN, CL_TRUE, 0, nodes.size() * real_num_node_sample() * sizeof(cl_double), randoms.data());

		// Randoms for update pi for each node
		int i = 0;
		for (auto node = nodes.begin();
				node != nodes.end();
				++node, ++i) {
			std::vector<double> noise = Random::random->randn(K);
			clContext.queue.enqueueWriteBuffer(clRandomNK, CL_TRUE,
					i * K * sizeof(cl_double),
					K * sizeof(cl_double),
					noise.data());
		}

		sample_latent_vars_and_update_pi_kernel.setArg(0, clGraph);
		sample_latent_vars_and_update_pi_kernel.setArg(1, clNodes);
		sample_latent_vars_and_update_pi_kernel.setArg(2, (cl_int)nodes.size());
		sample_latent_vars_and_update_pi_kernel.setArg(3, clNodesNeighbors);
		sample_latent_vars_and_update_pi_kernel.setArg(4, clPi);
		sample_latent_vars_and_update_pi_kernel.setArg(5, clPiUpdate);
		sample_latent_vars_and_update_pi_kernel.setArg(6, clPhi);
		sample_latent_vars_and_update_pi_kernel.setArg(7, clBeta);
		sample_latent_vars_and_update_pi_kernel.setArg(8, (cl_double)epsilon);
		sample_latent_vars_and_update_pi_kernel.setArg(9, clZ);
		sample_latent_vars_and_update_pi_kernel.setArg(10, clRandomNN);
		sample_latent_vars_and_update_pi_kernel.setArg(11, clRandomNK);
		sample_latent_vars_and_update_pi_kernel.setArg(12, clScratch);
		sample_latent_vars_and_update_pi_kernel.setArg(13, (cl_double)alpha);
		sample_latent_vars_and_update_pi_kernel.setArg(14, (cl_double)a);
		sample_latent_vars_and_update_pi_kernel.setArg(15, (cl_double)b);
		sample_latent_vars_and_update_pi_kernel.setArg(16, (cl_double)c);
		sample_latent_vars_and_update_pi_kernel.setArg(17, (cl_int)step_count);
		sample_latent_vars_and_update_pi_kernel.setArg(18, (cl_int)N);

		clContext.queue.enqueueNDRangeKernel(sample_latent_vars_and_update_pi_kernel, cl::NullRange, cl::NDRange(4), cl::NDRange(1));
		clContext.queue.finish();

		// read Pi again
		for (auto node = nodes.begin();
				node != nodes.end();
				++node) {
			clContext.queue.enqueueReadBuffer(clPiUpdate, CL_TRUE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					pi[*node].data());
			clContext.queue.enqueueCopyBuffer(clPiUpdate, clPi,
					*node * K * sizeof(cl_double),
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double));
		}
		clContext.queue.finish();
	}

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
	cl::Kernel sample_latent_vars_and_update_pi_kernel;
	cl::Kernel sample_latent_vars2_kernel;
	cl::Kernel update_beta_calculate_grads_kernel;
	cl::Kernel update_beta_calculate_theta_kernel;

	cl::Buffer clGraphEdges;
	cl::Buffer clGraphNodes;
	cl::Buffer clGraph;

	cl::Buffer clNodes;
	cl::Buffer clNodesNeighbors;
	cl::Buffer clPi;
	cl::Buffer clPiUpdate;
	cl::Buffer clPhi;
	cl::Buffer clBeta;
	cl::Buffer clTheta;
	cl::Buffer clThetaSum;
	cl::Buffer clZ;
	cl::Buffer clRandomNN;
	cl::Buffer clRandomNK;
	cl::Buffer clScratch;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
