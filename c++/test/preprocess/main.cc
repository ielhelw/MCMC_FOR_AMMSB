#include "mcmc/mcmc.h"

using namespace mcmc;
using namespace mcmc::preprocess;

int main(int argc, char *argv[]) {
	bool quiet;
	::size_t progress;

	boost::program_options::options_description options;
	options.add_options()
		("quiet,q",
		 po::bool_switch(&quiet)->default_value(false),
		 "quiet: no dump of data")
		("progress,p",
		 po::value<::size_t>(&progress)->default_value(0),
		 "progress: show progress every <progress> lines")
		;

  mcmc::Options mcmc_options(argc, argv, &options);

  DataFactory df(mcmc_options);
  df.setProgress(progress);

  const Data *data = df.get_data();
  if (! quiet) {
	  data->dump_data();
  }

  if (quiet) {
	  print_mem_usage(std::cerr);
	  std::cerr << "Hit <enter> to create Network" << std::endl;
	  std::string dummy;
	  getline(std::cin, dummy);
  }

  Network network(data, 0.1);

  std::cerr << "Network: N " << network.get_num_nodes() <<
	  " E " << network.get_num_total_edges() <<
	  " linked edges " << network.get_num_linked_edges() <<
	  " max.fan-out " << network.get_max_fan_out() <<
	  " held-out set " << network.get_held_out_set().size() <<
	  " test set " << network.get_test_set().size() <<
	  std::endl;

  if (quiet) {
	  print_mem_usage(std::cerr);
	  std::cerr << "Hit <enter> to quit" << std::endl;
	  std::string dummy;
	  getline(std::cin, dummy);
  }

  df.deleteData(data);

  return 0;
}
