#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;
using namespace mcmc::learning;

int main(int argc, char *argv[]) {
	try {
		Options args(argc, argv);

#ifdef ENABLE_OPENCL
		cl::ClContext context = cl::ClContext::createOpenCLContext(args.openClPlatform,
																   args.openClDevice);
#endif

		DataFactory df(args.dataset_class, args.filename);
		const Data *data = df.get_data();
		Network network(data, 0.1);

		if (args.run.mcmc_stochastical) {
			std::cout << "start MCMC stochastical" << std::endl;
			MCMCSamplerStochastic mcmcSampler(args, network);
			mcmcSampler.run();
		}

		if (args.run.mcmc_batch) {
			std::cout << "start MCMC batch" << std::endl;
			MCMCSamplerBatch mcmcSampler(args, network);
			mcmcSampler.run();
		}

#ifdef ENABLE_OPENCL
		if (args.run.mcmc_stochastical_cl) {
			std::cout << "start MCMC stochastical CL" << std::endl;
			MCMCClSamplerStochastic mcmcclSampler(args, network, context);
			mcmcclSampler.run();
		}
#ifdef IMPLEMENT_MCMC_CL_BATCH
		if (args.run.mcmc_batch_cl) {
			std::cout << "start MCMC batch CL" << std::endl;
			MCMCClSamplerBatch mcmcclSampler(args, network, context);
			mcmcclSampler.run();
		}
#endif
#endif

		if (false) {
			std::cout << "start variational inference batch" << std::endl;
			SV svSampler(args, network);
			svSampler.run();
		}

		if (false) {
			std::cout << "start variational inference stochastical" << std::endl;
			SVI sviSampler(args, network);
			sviSampler.run();
		}

		return 0;

	} catch (mcmc::IOException &e) {
		std::cerr << "IO error: " << e.what() << std::endl;
		return 33;

	} catch (boost::program_options::error &e) {
		std::cerr << "Option error: " << e.what() << std::endl;
		return 33;
	}
}
