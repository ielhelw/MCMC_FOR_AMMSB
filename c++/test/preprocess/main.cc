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
	double held_out_ratio;

	boost::program_options::options_description options;
	options.add_options()
		("quiet,q",
		 po::bool_switch(&quiet)->default_value(false),
		 "quiet: no dump of data")
		("save,O",
		 po::value<std::string>(&network_save),
		 "save network in some native format")
		("held-out-ratio,H",
		 po::value<double>(&held_out_ratio)->default_value(0.0),
		 "compress saved network file")
		;

	mcmc::Options mcmc_options(argc, argv, &options);

	auto start = system_clock::now();

	print_mem_usage(std::cerr);

	Network network;
	network.Init(mcmc_options, held_out_ratio);

	std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read Network" << std::endl;
	print_mem_usage(std::cerr);

	std::cerr << "Network: N " << network.get_num_nodes() <<
		" E " << network.get_num_linked_edges() <<
		" max.fan-out " << network.get_max_fan_out() <<
		" held-out set " << network.get_held_out_set().size() <<
		" test set " << network.get_test_set().size() <<
		std::endl;

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

	return 0;
}
