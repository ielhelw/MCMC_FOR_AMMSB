#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

#include <mcmc/random.h>


double stddev(double sum, double sumsq, ::size_t N) {
  double a = sumsq - sum * sum / N;
  if (a < 0.0) {
    std::cerr << std::setprecision(16) << "Oopppssss.... sum " << sum << " sum^2 " << (sum * sum) << " sum^2/N " << (sum * sum / N) << " sumsq " << sumsq << std::endl;
  }
  return sqrt((1.0 / (N - 1)) * (sumsq - sum * sum / N));
}

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

  auto rgen = mcmc::Random::Random(42);

  std::cout << "N " << N << " K " << K << std::endl;
#ifdef MCMC_RANDOM_SYSTEM
  std::cout << "RAND_MAX " << RAND_MAX << std::endl;
#endif
  std::cout << std::fixed;

  if (true) {
    double sum = 0.0;
    double sumsq = 0.0;
    auto start = std::chrono::system_clock::now();
    for (::size_t i = 0; i < N; i++) {
      auto r = rgen.randint(0, N);
      sum += r;
      sumsq += r * r;
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << N << " randint(" << N << ") takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / N) << std::endl;
    std::cout << "mean " << (sum / N) << " stdev " << stddev(sum, sumsq, N) << std::endl;
  }

  if (true) {
    double sum = 0.0;
    double sumsq = 0.0;
    auto start = std::chrono::system_clock::now();
    for (::size_t i = 0; i < N; i++) {
      auto r = rgen.random();
      sum += r;
      sumsq += r * r;
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << N << " random() takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / N) << std::endl;
    std::cout << "mean " << (sum / N) << " stdev " << stddev(sum, sumsq, N) << std::endl;
  }

  if (true) {
    double sum = 0.0;
    double sumsq = 0.0;
    std::vector<double> a(K);
    auto start = std::chrono::system_clock::now();
    for (::size_t i = 0; i < N / K; i++) {
      a = rgen.randn(K);
      for (auto r : a) {
        sum += r;
        sumsq += r * r;
      }
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << (N / K) << " randn(" << K << ") takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / ((N / K) * K)) << std::endl;
    std::cout << "mean " << (sum / N) << " stdev " << stddev(sum, sumsq, N) << std::endl;
    rgen.report();
  }

  if (true) {
    double sum = 0.0;
    double sumsq = 0.0;
    std::vector<std::vector<double> > a(N/K, std::vector<double>(K));
    auto start = std::chrono::system_clock::now();
    a = rgen.gamma(1.0, 1.0, N / K, K);
    for (auto b : a) {
      for (auto r : b) {
        sum += r;
        sumsq += r * r;
      }
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << "gamma(" << (N / K) << "," << K << ") takes total " << (stop - start).count() << " per call " << (1.0 * (stop - start).count() / ((N / K) * K)) << std::endl;
    std::cout << "mean " << (sum / N) << " stdev " << stddev(sum, sumsq, N) << std::endl;
    rgen.report();
  }

  return 0;
}
