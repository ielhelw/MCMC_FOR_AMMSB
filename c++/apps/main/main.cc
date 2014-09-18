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

		if (false) {
			std::cout << "start MCMC stochastical" << std::endl;
			MCMCSamplerStochastic mcmcSampler(args, network);
			mcmcSampler.run();
		}

		if (true) {
			std::cout << "start MCMC batch" << std::endl;
			MCMCSamplerBatch mcmcSampler(args, network);
			mcmcSampler.run();
		}

#ifdef ENABLE_OPENCL
		if (false) {
			std::cout << "start MCMC CL stochastical" << std::endl;
			MCMCClSamplerStochastic mcmcclSampler(args, network, context);
			mcmcclSampler.run();
		}
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

	} catch (boost::program_options::error &e) {
		std::cerr << "Option error: " << e.what() << std::endl;
		return 33;
	}
}
