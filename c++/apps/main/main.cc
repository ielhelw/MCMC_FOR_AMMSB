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
      // Parameter set for Python comparison:
      // -s -f ../../../../datasets/netscience.txt -c relativity -K 15 -m 147 -n 10 --mcmc.alpha 0.01 --mcmc.epsilon 0.0000001 --mcmc.held-out-ratio 0.009999999776 -i 1
      std::cout << "start MCMC stochastical" << std::endl;
      MCMCSamplerStochastic mcmcSampler(args);
      mcmcSampler.init();
      mcmcSampler.run();
    }

    if (args.run.mcmc_stochastical_distr) {
      // Parameter set for Python comparison:
      // -d -f ../../../../datasets/netscience.txt -c relativity -K 15 -m 147 -n 10 --mcmc.alpha 0.01 --mcmc.epsilon 0.0000001 --mcmc.held-out-ratio 0.009999999776
      std::cout << "start MCMC stochastical distributed " << std::endl;
      MCMCSamplerStochasticDistributed mcmcSampler(args);
      mcmcSampler.init();
      mcmcSampler.run();
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
