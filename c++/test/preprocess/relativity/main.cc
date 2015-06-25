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

  DataFactory df(mcmc_options.dataset_class, mcmc_options.filename);
  df.setCompressed(mcmc_options.compressed);
  df.setContiguous(mcmc_options.contiguous);
  df.setProgress(progress);

  const Data *data = df.get_data();
  if (! quiet) {
	  data->dump_data();
  }
  df.deleteData(data);

  return 0;
}
