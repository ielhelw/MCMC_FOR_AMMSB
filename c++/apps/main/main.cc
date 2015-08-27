#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;
using namespace mcmc::learning;

int main(int argc, char *argv[]) {
  namespace po = boost::program_options;
  try {
    std::cout << "Command line: ";
    for (int i = 0; i < argc; i++) {
      std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    po::options_description desc("Commandline options");
    desc.add_options()
      ("help", "help")
    ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv)
                                    .options(desc)
                                    .allow_unregistered()
                                    .run();
    po::store(parsed, vm);
    po::notify(vm);

    mcmc::Options args;
    if (vm.count("help") > 0) {
      std::cout << args << std::endl;
      DKV::DKVFile::DKVStoreFileOptions file_opts;
      std::cout << file_opts << std::endl;
#ifdef MCMC_ENABLE_RDMA
      DKV::DKVRDMA::DKVStoreRDMAOptions rdma_opts;
      std::cout << rdma_opts << std::endl;
#endif
#ifdef MCMC_ENABLE_RAMCLOUD
      DKV::DKVRamCloud::DKVStoreRamCloudOptions rc_opts;
      std::cout << rc_opts << std::endl;
#endif
      return 0;
    }
    auto remains = po::collect_unrecognized(parsed.options, po::include_positional);
    args.Parse(remains);

#ifdef MCMC_ENABLE_DISTRIBUTED
      // Parameter set for Python comparison:
      // -d -f ../../../../datasets/netscience.txt -c relativity -K 15 -m 147 -n 10 --mcmc.alpha 0.01 --mcmc.epsilon 0.0000001 --mcmc.held-out-ratio 0.009999999776
      std::cout << "start MCMC stochastical distributed " << std::endl;
      MCMCSamplerStochasticDistributed mcmcSampler(args);
#else
      // Parameter set for Python comparison:
      // -s -f ../../../../datasets/netscience.txt -c relativity -K 15 -m 147 -n 10 --mcmc.alpha 0.01 --mcmc.epsilon 0.0000001 --mcmc.held-out-ratio 0.009999999776 -i 1
      std::cout << "start MCMC stochastical" << std::endl;
      MCMCSamplerStochastic mcmcSampler(args);
#endif
      mcmcSampler.init();
      mcmcSampler.run();

    return 0;
  } catch (mcmc::IOException &e) {
    std::cerr << "IO error: " << e.what() << std::endl;
    return 33;

  } catch (boost::program_options::error &e) {
    std::cerr << "Option error: " << e.what() << std::endl;
    return 33;
  }
}
