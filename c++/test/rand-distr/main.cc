#include <chrono>
#include <iostream>

#include <mcmc/options.h>
#include <mcmc/random.h>


int main(int argc, char *argv[]) {
	uint64_t N;
	uint64_t K;
	mcmc::Float eta[2];

	namespace po = ::boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("eta0,0", po::value<mcmc::Float>(&eta[0])->default_value(1.0), "eta0")
		("eta1,1", po::value<mcmc::Float>(&eta[1])->default_value(1.0), "eta0")
		("K,K", po::value<uint64_t>(&K)->default_value(300), "sample batch size")
		("iterations,N", po::value<uint64_t>(&N)->default_value(1 << 28), "iterations")
		;

	po::positional_options_description p;
	p.add("iterations", -1);
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.count("iterations") > 0) {
		N = vm["iterations"].as<uint64_t>();
	}

	std::cout << "N " << N << std::endl;

	::size_t n = N * K;

        auto rgen = mcmc::Random::Random(42);

	if (true) {
		mcmc::Float sum = 0.0;
		mcmc::Float sq  = 0.0;

		auto start = std::chrono::system_clock::now();
		for (::size_t i = 0; i < N; i++) {
			std::vector<std::vector<mcmc::Float> > r = rgen.gamma(eta[0], eta[1], 1, K);
			for (::size_t k = 0; k < K; k++) {
				sum += r[0][k];
				sq  += r[0][k] * r[0][k];
			}
		}

		auto stop = std::chrono::system_clock::now();
		std::cout << n << " random->gamma() takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / n) << std::endl;

		std::cout << "Random gamma mean " << (sum / n) << " s.dev. " << std::sqrt((sq - sum * sum / n) / (n - 1)) << std::endl;
	}

	if (true) {
		mcmc::Float sum = 0.0;
		mcmc::Float sq  = 0.0;

		auto start = std::chrono::system_clock::now();
		for (::size_t i = 0; i < N; i++) {
			std::vector<mcmc::Float> r = rgen.randn(K);
			for (::size_t k = 0; k < K; k++) {
				sum += r[k];
				sq  += r[k] * r[k];
			}
		}

		auto stop = std::chrono::system_clock::now();
		std::cout << n << " random->randn() takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / n) << std::endl;

		std::cout << "Random randn mean " << (sum / n) << " s.dev. " << std::sqrt((sq - sum * sum / n) / (n - 1)) << std::endl;
	}

	return 0;
}
