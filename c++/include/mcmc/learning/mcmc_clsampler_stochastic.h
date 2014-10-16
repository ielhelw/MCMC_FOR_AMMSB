#ifndef MCMC_CLSAMPLER_STOCHASTIC_H_
#define MCMC_CLSAMPLER_STOCHASTIC_H_

#include <memory>
#include <algorithm>
#include <chrono>

#include "mcmc_clsampler.h"

#include "opencl/context.h"

#define PARALLELISM 1

namespace mcmc {
namespace learning {

class MCMCClSamplerStochastic : public MCMCClSampler {
public:
	MCMCClSamplerStochastic(const Options &args, const Network &network, const cl::ClContext clContext, double eta0 = 100.0, double eta1 = 0.01)
		: MCMCClSampler(args, network, network.get_num_nodes() / 5, eta0, eta1, clContext) {

		sampler_program = this->clContext.createProgram(stringify(PROJECT_HOME) "/../OpenCL/mcmc_sampler_stochastic.cl", progOpts);
		sample_latent_vars_and_update_pi_kernel = cl::Kernel(sampler_program, "sample_latent_vars_and_update_pi");
		sample_latent_vars2_kernel = cl::Kernel(sampler_program, "sample_latent_vars2");
		update_beta_calculate_grads_kernel = cl::Kernel(sampler_program, "update_beta_calculate_grads");
		update_beta_calculate_theta_kernel = cl::Kernel(sampler_program, "update_beta_calculate_theta");

		info(std::cout);
	}

	virtual void run() {
		/** run mini-batch based MCMC sampler, based on the sungjin's note */

		if (step_count % 1 == 0) {
			double ppx_score = cal_perplexity_held_out();
			std::cout << std::fixed << std::setprecision(15) << "perplexity for hold out set is: " << ppx_score << std::endl;
			ppxs_held_out.push_back(ppx_score);
		}

		while (step_count < max_iteration && ! is_converged()) {
			auto l1 = std::chrono::system_clock::now();

			// (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
			EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
			const OrderedEdgeSet &mini_batch = *edgeSample.first;
			double scale = edgeSample.second;

			// iterate through each node in the mini batch.
			OrderedVertexSet nodes = nodes_in_batch(mini_batch);

			sample_latent_vars_and_update_pi(nodes);

			// sample (z_ab, z_ba) for each edge in the mini_batch.
			// z is map structure. i.e  z = {(1,10):3, (2,4):-1}
			sample_latent_vars2(mini_batch);

			update_beta(mini_batch, scale);


			if (step_count % 1 == 0) {
				double ppx_score = cal_perplexity_held_out();
				std::cout << std::fixed << std::setprecision(12) << "perplexity for hold out set is: " << ppx_score << std::endl;
				ppxs_held_out.push_back(ppx_score);
			}

			delete edgeSample.first;

			step_count++;
			auto l2 = std::chrono::system_clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(l2 - l1);
			std::cout << "LOOP  = " << ms.count() << "ms" << std::endl;
		}
	}

protected:

	void update_beta(const OrderedEdgeSet &mini_batch, double scale) {
		// copy theta_sum
		std::vector<cl_double> vThetaSum(theta.size());
		std::transform(theta.begin(), theta.end(), vThetaSum.begin(), np::sum<double>);
		clContext.queue.enqueueWriteBuffer(clThetaSum, CL_FALSE,
				0, theta.size() * sizeof(cl_double),
				vThetaSum.data());

		update_beta_calculate_grads_kernel.setArg(0, clGraph);
		update_beta_calculate_grads_kernel.setArg(1, clEdges);
		update_beta_calculate_grads_kernel.setArg(2, (cl_int)mini_batch.size());
		update_beta_calculate_grads_kernel.setArg(3, clZ);
		update_beta_calculate_grads_kernel.setArg(4, clTheta);
		update_beta_calculate_grads_kernel.setArg(5, clThetaSum);
		update_beta_calculate_grads_kernel.setArg(6, clScratch);
		update_beta_calculate_grads_kernel.setArg(7, (cl_double)scale);

		::size_t countPartialSums = std::min(mini_batch.size(), (::size_t)PARALLELISM);

		clContext.queue.finish(); // Wait for sample_latent_vars2

		cl::Event e_grads_kernel;
		clContext.queue.enqueueNDRangeKernel(update_beta_calculate_grads_kernel, cl::NullRange,
				cl::NDRange(countPartialSums), cl::NDRange(1),
				NULL, &e_grads_kernel);

		double eps_t = a * std::pow(1.0 + step_count / b, -c);
		cl_double2 clEta;
		clEta.s[0] = this->eta[0];
		clEta.s[1] = this->eta[1];

		e_grads_kernel.wait();

		update_beta_calculate_theta_kernel.setArg(0, clTheta);
		update_beta_calculate_theta_kernel.setArg(1, clRandomSeed);
		update_beta_calculate_theta_kernel.setArg(2, clScratch);
		update_beta_calculate_theta_kernel.setArg(3, (cl_double)scale);
		update_beta_calculate_theta_kernel.setArg(4, (cl_double)eps_t);
		update_beta_calculate_theta_kernel.setArg(5, clEta);
		update_beta_calculate_theta_kernel.setArg(6, (int)countPartialSums);

		clContext.queue.enqueueTask(update_beta_calculate_theta_kernel);
		clContext.queue.finish();


		// copy theta
		std::vector<cl_double2> vclTheta(theta.size());
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

	void sample_latent_vars2(const OrderedEdgeSet &mini_batch) {
		std::vector<cl_int2> edges(mini_batch.size());
		// Copy edges
		std::transform(mini_batch.begin(), mini_batch.end(), edges.begin(), [](const Edge& e) {
			cl_int2 E;
			E.s[0] = e.first;
			E.s[1] = e.second;
			return E;
		});
		clContext.queue.enqueueWriteBuffer(clEdges, CL_FALSE,
				0, edges.size()*sizeof(cl_int2),
				edges.data());

		sample_latent_vars2_kernel.setArg(0, clGraph);
		sample_latent_vars2_kernel.setArg(1, clEdges);
		sample_latent_vars2_kernel.setArg(2, (cl_int)edges.size());
		sample_latent_vars2_kernel.setArg(3, clPi);
		sample_latent_vars2_kernel.setArg(4, clBeta);
		sample_latent_vars2_kernel.setArg(5, clZ);
		sample_latent_vars2_kernel.setArg(6, clScratch);
		sample_latent_vars2_kernel.setArg(7, clRandomSeed);

		clContext.queue.finish(); // Wait for clEdges and PiUpdates from sample_latent_vars_and_update_pi

		clContext.queue.enqueueNDRangeKernel(sample_latent_vars2_kernel, cl::NullRange, cl::NDRange(PARALLELISM), cl::NDRange(1));
	}

	::size_t real_num_node_sample() const {
		return num_node_sample + 1;
	}

	void sample_latent_vars_and_update_pi(const OrderedVertexSet& nodes) {

		// Copy sampled node IDs
		std::vector<int> v_nodes(nodes.begin(), nodes.end()); // FIXME: replace OrderedVertexSet with vector
		clContext.queue.enqueueWriteBuffer(clNodes, CL_FALSE, 0, v_nodes.size()*sizeof(int), v_nodes.data());

		// Copy beta
		clContext.queue.enqueueWriteBuffer(clBeta, CL_FALSE, 0, K * sizeof(double), beta.data());

		int Idx = 0;
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clGraph);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clHeldOutGraph);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clNodes);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_int)nodes.size());
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clNodesNeighbors);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clPi);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clPiUpdate);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clPhi);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clBeta);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_double)epsilon);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clZ);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clScratch);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_double)alpha);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_double)a);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_double)b);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_double)c);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_int)step_count);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, (cl_int)N);
		sample_latent_vars_and_update_pi_kernel.setArg(Idx++, clRandomSeed);

		clContext.queue.finish();
		clContext.queue.enqueueNDRangeKernel(sample_latent_vars_and_update_pi_kernel, cl::NullRange, cl::NDRange(PARALLELISM), cl::NDRange(1));
		clContext.queue.finish();

		// read Pi again
		for (auto node = nodes.begin();
				node != nodes.end();
				++node) {
			clContext.queue.enqueueReadBuffer(clPiUpdate, CL_FALSE,
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double),
					pi[*node].data());
			clContext.queue.enqueueCopyBuffer(clPiUpdate, clPi,
					*node * K * sizeof(cl_double),
					*node * K * sizeof(cl_double),
					K * sizeof(cl_double));
		}
	}

	cl::Program sampler_program;

	cl::Kernel sample_latent_vars_and_update_pi_kernel;
	cl::Kernel sample_latent_vars2_kernel;
	cl::Kernel update_beta_calculate_grads_kernel;
	cl::Kernel update_beta_calculate_theta_kernel;
};

}
}


#endif /* MCMC_CLSAMPLER_STOCHASTIC_H_ */
