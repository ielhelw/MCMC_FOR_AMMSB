#include <chrono>
#include <iostream>

#include <mcmc/options.h>
#include <mcmc/random.h>


int main(int argc, char *argv[]) {
	uint64_t N;
	uint64_t K;

	namespace po = ::boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("K,K", po::value<uint64_t>(&K)->default_value(300), "K")
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

	std::cout << "N " << N << " K " << K << std::endl;

	if (true) {
		auto start = std::chrono::system_clock::now();
		for (::size_t i = 0; i < N; i++) {
			(void)mcmc::Random::random->randint(0, N);
		}
		auto stop = std::chrono::system_clock::now();
		std::cout << N << " randint() takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / N) << std::endl;
	}

	if (true) {
		auto start = std::chrono::system_clock::now();
		for (::size_t i = 0; i < N; i++) {
			(void)mcmc::Random::random->random();
		}
		auto stop = std::chrono::system_clock::now();
		std::cout << N << " random() takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / N) << std::endl;
	}

	if (true) {
		std::vector<double> a(K);
		auto start = std::chrono::system_clock::now();
		for (::size_t i = 0; i < N / K; i++) {
			a = mcmc::Random::random->randn(K);
		}
		auto stop = std::chrono::system_clock::now();
		std::cout << (N / K) << " randn(" << K << ") takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / ((N / K) * K)) << std::endl;
		mcmc::Random::random->report();
	}

	return 0;
}
