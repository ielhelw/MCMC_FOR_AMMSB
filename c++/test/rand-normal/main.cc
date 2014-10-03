#include <chrono>
#include <iostream>

#include <mcmc/random.h>


int main(int argc, char *argv[]) {
	const ::size_t N = 1 << 25;
	const ::size_t K = 300;

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
	}

	return 0;
}
