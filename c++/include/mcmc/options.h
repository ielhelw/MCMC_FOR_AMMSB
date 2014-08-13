#ifndef MCMC_OPTIONS_H__
#define MCMC_OPTIONS_H__

#include <boost/program_options.hpp>


namespace mcmc {

namespace po = ::boost::program_options;

class Options {
public:
	Options(int argc, char *argv[]) {
		po::options_description desc("Options");

		desc.add_options()
			("help,?", "help")
			("alpha", po::value<double>(&alpha)->default_value(0.01), "alpha")
			("eta0", po::value<double>(&eta0)->default_value(1.0), "eta0")
			("eta1", po::value<double>(&eta1)->default_value(1.0), "eta1")

			("K,k", po::value<double>(&K)->default_value(300), "K")
			("mini-batch-size,b", po::value<double>(&mini_batch_size)->default_value(50), "mini_batch_size")

			("epsilon,e", po::value<double>(&epsilon)->default_value(0.05), "epsilon")
			("max-iteration,x", po::value<double>(&max_iteration)->default_value(10000000), "max_iteration")

			("a", po::value<double>(&a)->default_value(0.01), "a")
			("b", po::value<double>(&b)->default_value(1024), "b")
			("c", po::value<double>(&c)->default_value(0.55), "c")

			("num-updates,u", po::value<double>(&num_updates)->default_value(1000), "num_updates")
			("hold-out-prob,h", po::value<double>(&hold_out_prob)->default_value(0.1), "hold_out_prob")
			("output-dir,o", po::value<double>(&output_dir)->default_value("."), "output_dir")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help") > 0) {
			std::cout << desc << std::endl;
		}
	}

public:
	double		alpha;
	double		eta0;
	double		eta1;
	::size_t	K;
	::size_t	mini_batch_size;
	double		epsilon;
	::size_t	max_iteration;

	// parameters for step size
	double		a;
	double		b;
	double		c;

	::size_t	num_updates;
	double		hold_out_prob;
	std::string	output_dir;
};

};	// namespace mcmc

#endif	// ndef MCMC_OPTIONS_H__
