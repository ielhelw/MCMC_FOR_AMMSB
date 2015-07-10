#include "mcmc/options.h"
#include "mcmc/preprocess/data_factory.h"
#include "mcmc/network.h"

using namespace mcmc;
using namespace mcmc::preprocess;

int main(int argc, char *argv[]) {
	bool quiet;
	::size_t progress;
	std::string network_save;
	bool compressed;

	boost::program_options::options_description options;
	options.add_options()
		("quiet,q",
		 po::bool_switch(&quiet)->default_value(false),
		 "quiet: no dump of data")
		("progress,p",
		 po::value<::size_t>(&progress)->default_value(0),
		 "progress: show progress every <progress> lines")
		("save,O",
		 po::value<std::string>(&network_save),
		 "save network in some native format")
		("compress,Z",
		 po::bool_switch(&compressed)->default_value(false),
		 "compress saved network file")
		;

	mcmc::Options mcmc_options(argc, argv, &options);

	DataFactory df(mcmc_options);
	Network network;
  const Data *data = NULL;
  if (mcmc_options.dataset_class == "gz" ||
		  mcmc_options.dataset_class == "sparsehash") {
	  network = Network(mcmc_options.filename, mcmc_options.compressed);
	  network.Init(0.1);
  } else {
	  df.setProgress(progress);

	  data = df.get_data();
	  if (! quiet) {
		  data->dump_data();
	  }

	  if (quiet) {
		  print_mem_usage(std::cerr);
		  std::cerr << "Hit <enter> to create Network" << std::endl;
		  std::string dummy;
		  getline(std::cin, dummy);
	  }

	  network.Init(data, 0.1);
  }

  std::cerr << "Network: N " << network.get_num_nodes() <<
	  " E " << network.get_num_linked_edges() <<
	  " max.fan-out " << network.get_max_fan_out() <<
	  " held-out set " << network.get_held_out_set().size() <<
	  " test set " << network.get_test_set().size() <<
	  std::endl;

  if (network_save != "") {
	  if (quiet) {
		  print_mem_usage(std::cerr);
		  std::cerr << "Hit <enter> to save" << std::endl;
		  std::string dummy;
		  getline(std::cin, dummy);
	  }

	  FILE *os;
	  if (compressed) {
		  std::string cmd("gzip > " + network_save);
		  os = popen(cmd.c_str(), "w");
		  if (os == NULL) {
			  throw mcmc::MCMCException("Cannot popen(" + cmd + ")");
		  }
	  } else {
		  os = fopen(network_save.c_str(), "w");
		  if (os == NULL) {
			  throw mcmc::MCMCException("Cannot write open/w file" + network_save);
		  }
	  }

	  int32_t N = network.get_num_nodes();
	  if (fwrite(&N, sizeof N, 1, os) == 0) {
		  throw mcmc::MCMCException("Cannot write #nodes to file");
	  }
	  for (auto r : network.get_linked_edges()) {
		  GoogleHashSet &rc = const_cast<GoogleHashSet &>(r);
		  rc.write_metadata(os);
		  rc.write_nopointer_data(os);
	  }

	  GoogleHashMap &hc = const_cast<GoogleHashMap &>(network.get_held_out_set());
	  hc.write_metadata(os);
	  hc.write_nopointer_data(os);
	  GoogleHashMap &tc = const_cast<GoogleHashMap &>(network.get_test_set());
	  tc.write_metadata(os);
	  tc.write_nopointer_data(os);

	  if (compressed) {
		  if (os != NULL) {
			  pclose(os);
		  }
	  } else {
		  if (os != NULL) {
			  fclose(os);
		  }
	  }
  }

  if (quiet) {
	  print_mem_usage(std::cerr);
	  std::cerr << "Hit <enter> to quit" << std::endl;
	  std::string dummy;
	  getline(std::cin, dummy);
  }

  df.deleteData(data);

  return 0;
}
