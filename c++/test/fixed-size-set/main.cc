#include <iostream>
#include <cstdlib>

#include <mcmc/fixed-size-set.h>

int main(int argc, char *argv[]) {
  ::size_t N = 32;
  mcmc::FixedSizeSet set(N);

  srandom(42);
  for (::size_t i = 0; i < N; ++i) {
    int r = random();
    set.insert(r);
  }

  std::cout << "Contains ";
  for (auto r = set.begin(); r != set.end(); ++r) {
    std::cout << *r << " ";
  }
  std::cout << std::endl;

  while (true) {
    std::cout << "> ";
    std::flush(std::cout);
    int r;
    std::cin >> r;
    std::cout << r << " is ";
    bool is_in = set.find(r) != set.end();
    if (! is_in) {
      std::cout << "not ";
    }
    std::cout << "in the set" << std::endl;
  }

  return 0;
}
