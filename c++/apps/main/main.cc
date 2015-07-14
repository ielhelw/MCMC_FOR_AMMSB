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

    if (args.run.mcmc_stochastical_distr) {
      std::cout << "start MCMC stochastical distributed " << std::endl;
      MCMCSamplerStochasticDistributed mcmcSampler(args);
      mcmcSampler.init();
      mcmcSampler.run();
    }

    return 0;
#ifdef ENABLE_OPENCL
  } catch (cl::Error &e) {
    std::cerr << "OpenCL error [code=" << e.err() << "]: " << e.what()
              << std::endl;
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
