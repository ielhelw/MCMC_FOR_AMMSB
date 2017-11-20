#include <iostream>

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#endif
#include <boost/program_options.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif

#include <mcmc/timer.h>
#include <mr/timer.h>

#include <mcmc/options.h>

template <typename timer>
class Tester {
 public:
  void run(::size_t n, const std::string &name) {
    timer::setTabular(true);

    timer t_outer(name + "Outer time");
    timer t_test("test timer");

    t_outer.start();
    for (::size_t i = 0; i < n; ++i) {
      t_test.start();
      t_test.stop();
    }
    t_outer.stop();

    timer::printHeader(std::cout);
    std::cout << t_test << std::endl;
    std::cout << t_outer << std::endl;
  }

};


int main(int argc, char *argv[]) {
  namespace po = boost::program_options;

  ::size_t n;

  po::options_description desc("Timer test");
  desc.add_options()
    ("help,?", "help")
    ("N,n", po::value<std::size_t>(&n)->default_value(1U << 20), "test size")
    ;

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl;
    return 33;
  }

  Tester<mcmc::timer::Timer> test_mcmc;
  test_mcmc.run(n, "MCMC");
  Tester<mr::timer::Timer>   test_mr;
  test_mr.run(n, "MR");

  return 0;
}
