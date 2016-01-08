#include <chrono>

#include "mcmc/options.h"
#include "mcmc/preprocess/data_factory.h"
#include "mcmc/network.h"

using namespace std::chrono;
using namespace mcmc;
using namespace mcmc::preprocess;

int main(int argc, char *argv[]) {
  bool quiet;
  std::string network_save;

  boost::program_options::options_description options;
  options.add_options()
    ("quiet,q",
     po::bool_switch(&quiet)->default_value(false),
     "quiet: no dump of data")
    ("save,O",
     po::value<std::string>(&network_save),
     "save network in some native format")
    ;
  po::variables_map vm;
  po::parsed_options parsed =
    po::basic_command_line_parser<char>(argc, argv).options(options)
    .allow_unregistered().run();
  po::store(parsed, vm);
  po::notify(vm);

  std::vector<std::string> remains = po::collect_unrecognized(
    parsed.options, po::include_positional);

  mcmc::Options mcmc_options(remains);

  auto start = system_clock::now();

  print_mem_usage(std::cerr);

  std::vector<Random::Random *> rng(omp_get_max_threads());
  for (auto & r : rng) {
    r = new Random::Random(mcmc_options.random_seed);
  }

  Network network;
  network.Init(mcmc_options, mcmc_options.held_out_ratio, &rng);

  std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read Network" << std::endl;
  print_mem_usage(std::cerr);

  const Data *data = network.get_data();
  if (! quiet) {
    data->dump_data();
    std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms dumped Network" << std::endl;
  }

  if (network_save != "") {
    print_mem_usage(std::cerr);
    network.save(network_save);
    std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms saved preprocessed Network etc" << std::endl;
  }

  std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms done" << std::endl;
  print_mem_usage(std::cerr);

  for (auto & r : rng) {
    delete r;
  }

	return 0;
}
