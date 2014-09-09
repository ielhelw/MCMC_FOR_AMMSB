#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;
using namespace mcmc::learning;

int main(int argc, char *argv[]) {
	Options args(argc, argv);

	DataFactory df(args.dataset_class, args.filename);
	const Data *data = df.get_data();
	Network network(data, 0.1);

	if (true) {
		std::cout << "start MCMC batch" << std::endl;
		MCMCSamplerBatch mcmcSampler(args, network);
		mcmcSampler.run();
	}

	if (false) {
		std::cout << "start MCMC stochastical" << std::endl;
		MCMCSamplerStochastic mcmcSampler(args, network);
		mcmcSampler.run();
	}

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
}
