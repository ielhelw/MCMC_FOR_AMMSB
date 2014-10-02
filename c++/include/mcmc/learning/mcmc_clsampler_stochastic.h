#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>
#include <chrono>

#include "mcmc_sampler_stochastic.h"
#include "mcmc_clsampler.h"

#include "opencl/context.h"


namespace mcmc {
namespace learning {

class MCMCClSamplerStochastic : public MCMCSamplerStochastic, public MCMCClSampler {
public:
	MCMCClSamplerStochastic(const Options &args, const Network &network, const cl::ClContext clContext, double eta0 = 100.0, double eta1 = 0.01)
		: Learner(args, network), MCMCSampler(args, network, N / 5, eta0, eta1), MCMCSamplerStochastic(args, network), MCMCClSampler(args, network, N / 5, eta0, eta1, clContext) {

		sampler_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/mcmc_sampler_stochastic.cl", progOpts);
		sample_latent_vars_kernel = cl::Kernel(sampler_program, "sample_latent_vars");
		update_pi_kernel = cl::Kernel(sampler_program, "update_pi_for_node");
		sample_latent_vars2_kernel = cl::Kernel(sampler_program, "sample_latent_vars2");

		info(std::cout);
	}

protected:

	virtual EdgeMapZ sample_latent_vars2(const OrderedEdgeSet &mini_batch) {
		std::vector<cl_int2> edges(mini_batch.size());
		int i = 0;
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
		clContext.queue.enqueueWriteBuffer(clRandom, CL_TRUE,
				0, edges.size() * sizeof(cl_double),
				randoms.data());

		sample_latent_vars2_kernel.setArg(0, clGraph);
		sample_latent_vars2_kernel.setArg(1, clNodesNeighbors);
		sample_latent_vars2_kernel.setArg(2, (cl_int)edges.size());
		sample_latent_vars2_kernel.setArg(3, clPi);
		sample_latent_vars2_kernel.setArg(4, clBeta);
		sample_latent_vars2_kernel.setArg(5, clZ);
		sample_latent_vars2_kernel.setArg(6, clScratch);
		sample_latent_vars2_kernel.setArg(7, clRandom);

		clContext.queue.enqueueNDRangeKernel(sample_latent_vars2_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
		clContext.queue.finish();

		std::vector<int> zFromCL(edges.size());
		clContext.queue.enqueueReadBuffer(clZ, CL_TRUE,
				0, edges.size() * sizeof(cl_int),
				zFromCL.data());
		EdgeMapZ ezm;
		i = 0;
		for (auto &e : mini_batch) {
			ezm[e] = zFromCL[i];
			++i;
		}
		return ezm;
	}

	virtual void update_pi_for_node_stub(const OrderedVertexSet& nodes,
			std::unordered_map<int, ::size_t>& size,
			std::unordered_map<int, std::vector<int> >& latent_vars,
			double scale) {
		// update pi for each node
		int i = 0;
		for (auto node = nodes.begin();
				node != nodes.end();
				++node, ++i) {
			std::vector<double> noise = Random::random->randn(K);
			clContext.queue.enqueueWriteBuffer(clRandom, CL_TRUE,
					i * K * sizeof(cl_double),
					K * sizeof(cl_double),
					noise.data());
			clContext.queue.enqueueWriteBuffer(clPhi, CL_TRUE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					phi[*node].data());
		}
		update_pi_kernel.setArg(0, clNodes);
		update_pi_kernel.setArg(1, (cl_int)nodes.size());
		update_pi_kernel.setArg(2, clPi);
		update_pi_kernel.setArg(3, clPhi);
		update_pi_kernel.setArg(4, clZ);
		update_pi_kernel.setArg(5, clRandom);
		update_pi_kernel.setArg(6, clScratch);
		update_pi_kernel.setArg(7, (cl_double)alpha);
		update_pi_kernel.setArg(8, (cl_double)a);
		update_pi_kernel.setArg(9, (cl_double)b);
		update_pi_kernel.setArg(10, (cl_double)c);
		update_pi_kernel.setArg(11, (cl_int)step_count);
		update_pi_kernel.setArg(12, (cl_int)N);

		// FIXME: threading granularity
		clContext.queue.enqueueNDRangeKernel(update_pi_kernel, cl::NullRange, cl::NDRange(4), cl::NDRange(1));
		clContext.queue.finish();

		// read Pi again
		for (auto node = nodes.begin();
				node != nodes.end();
				++node) {
			clContext.queue.enqueueReadBuffer(clPi, CL_TRUE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					pi[*node].data());
			clContext.queue.enqueueReadBuffer(clPhi, CL_TRUE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					phi[*node].data());
		}
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
		clContext.queue.enqueueWriteBuffer(clRandom, CL_TRUE, 0, nodes.size() * real_num_node_sample() * sizeof(cl_double), randoms.data());

		sample_latent_vars_kernel.setArg(0, clGraph);
		sample_latent_vars_kernel.setArg(1, clNodes);
		sample_latent_vars_kernel.setArg(2, (cl_int)nodes.size());
		sample_latent_vars_kernel.setArg(3, clNodesNeighbors);
		sample_latent_vars_kernel.setArg(4, clPi);
		sample_latent_vars_kernel.setArg(5, clBeta);
		sample_latent_vars_kernel.setArg(6, (cl_double)epsilon);
		sample_latent_vars_kernel.setArg(7, clZ);
		sample_latent_vars_kernel.setArg(8, clRandom);
		sample_latent_vars_kernel.setArg(9, clScratch);

		// FIXME: threading granularity
		clContext.queue.enqueueNDRangeKernel(sample_latent_vars_kernel, cl::NullRange, cl::NDRange(4), cl::NDRange(1));
		clContext.queue.finish();

		for (auto node = nodes.begin(); node != nodes.end(); ++node) {
			latent_vars[*node] = std::vector<int>(K, 0);
			clContext.queue.enqueueReadBuffer(clZ, CL_TRUE,
					(*node) * K * sizeof(cl_int),
					K * sizeof(cl_int),
					latent_vars[*node].data());
		}
	}

	cl::Program sampler_program;
	cl::Kernel sample_latent_vars_kernel;
	cl::Kernel update_pi_kernel;
	cl::Kernel sample_latent_vars2_kernel;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
