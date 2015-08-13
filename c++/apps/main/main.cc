
#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;
using namespace mcmc::learning;

int main(int argc, char *argv[]) {
	try {
		std::cout << "Command line: ";
		for (int i = 0; i < argc; i++) {
			std::cout << argv[i] << " ";
		}
		std::cout << std::endl;

        mcmc::Options args(argc, argv);

#ifdef ENABLE_OPENCL
		cl::ClContext context = cl::ClContext::createOpenCLContext(args.openClPlatform,
																   args.openClDevice);
#endif
		if (! args.run.mcmc_stochastical
			&& ! args.run.mcmc_batch
#ifdef ENABLE_OPENCL
			&& ! args.run.mcmc_stochastical_cl
#ifdef IMPLEMENT_MCMC_CL_BATCH
			&& ! args.run.mcmc_batch_cl
#endif
#endif
			&& ! args.run.mcmc_stochastical_distr
			) {
			std::cerr << "No compute device selected. Is that what you wanted?" << std::endl;
		}

		if (args.run.mcmc_stochastical) {
			std::cout << "start MCMC stochastical" << std::endl;
			MCMCSamplerStochastic mcmcSampler(args);
			mcmcSampler.init();
			mcmcSampler.run();
		}

		if (args.run.mcmc_batch) {
			std::cout << "start MCMC batch" << std::endl;
			MCMCSamplerBatch mcmcSampler(args);
			mcmcSampler.run();
		}

#ifdef ENABLE_OPENCL
		if (args.run.mcmc_stochastical_cl) {
			std::cout << "start MCMC stochastical CL" << std::endl;
			MCMCClSamplerStochastic mcmcclSampler(args, context);
			mcmcclSampler.run();
		}
#ifdef IMPLEMENT_MCMC_CL_BATCH
		if (args.run.mcmc_batch_cl) {
			std::cout << "start MCMC batch CL" << std::endl;
			MCMCClSamplerBatch mcmcclSampler(args, context);
			mcmcclSampler.run();
		}
#endif
#endif

		if (args.run.mcmc_stochastical_distr) {
			std::cout << "start MCMC stochastical distributed " << std::endl;
			MCMCSamplerStochasticDistributed mcmcSampler(args);
			mcmcSampler.init();
			mcmcSampler.run();
		}

		if (false) {
			std::cout << "start variational inference batch" << std::endl;
			SV svSampler(args);
			svSampler.run();
		}

		if (false) {
			std::cout << "start variational inference stochastical" << std::endl;
			SVI sviSampler(args);
			sviSampler.run();
		}

		return 0;
#ifdef ENABLE_OPENCL
	} catch (cl::Error &e) {
		std::cerr << "OpenCL error [code=" << e.err() << "]: " << e.what() << std::endl;
		return 33;
#endif
	} catch (mcmc::IOException &e) {
		std::cerr << "IO error: " << e.what() << std::endl;
		return 33;

	} catch (boost::program_options::error &e) {
		std::cerr << "Option error: " << e.what() << std::endl;
		return 33;
	}
}
