#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <string>
#include <set>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "mcmc/config.h"
#include "mcmc/exception.h"

namespace mcmc {
namespace Random {

class Random {
 protected:
  Random();

 public:
  Random(unsigned int seed, bool preserve_range_order = false);

  Random(unsigned int seed_hi, unsigned int seed_lo, bool preserve_range_order = false);

  ~Random();

#ifndef MCMC_RANDOM_SYSTEM
  inline uint64_t xorshift_128plus();

  inline uint64_t rand();
#endif

  uint64_t seed(int x) const;

  int64_t randint(int64_t from, int64_t upto);

  double random();

  double randn();
  std::vector<double> randn(::size_t K);
  std::vector<std::vector<double> > randn(::size_t K, ::size_t N);

  template <class List>
  List *sample(const List &population, ::size_t count);

  template <class Element>
  std::vector<Element> *sample(const std::vector<Element> &population,
                               ::size_t count);

  std::vector<int> *sampleRange(int N, ::size_t count);

  template <class Container>
  std::list<typename Container::value_type> *sampleList(
      const Container &population, ::size_t count);

  double gamma(double p1, double p2);
  std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1,
                                          ::size_t n2);

  void report();

  std::string state();

 protected:
  std::unordered_set<int> sample(int from, int upto, ::size_t count);

  template <class Input, class Result, class Inserter>
  void sample(Result *result, const Input &input, ::size_t count,
              Inserter inserter);

#ifndef MCMC_RANDOM_SYSTEM
  double gsl_ran_gaussian_ziggurat(const double sigma);

  double gsl_rng_uniform_pos();

  double gsl_ran_gamma(double a, double b);

  uint64_t xorshift_state[2];

  ::size_t n_randn = 0;
  ::size_t iters_randn = 0;
  ::size_t log_exp_randn = 0;
  ::size_t ktab_exceed_randn = 0;

#else
  std::default_random_engine generator;
  std::normal_distribution<double> normalDistribution;
#endif  // ndef MCMC_RANDOM_SYSTEM

private:
  bool preserve_range_order_;
};

// Random
template <class Input, class Result, class Inserter>
void Random::sample(Result *result, const Input &input, ::size_t count,
                    Inserter inserter) {
  std::unordered_set<int> accu = sample(0, (int)input.size(), count);

  ::size_t c = 0;
  for (auto i : input) {
    if (accu.find(c) != accu.end()) {
      inserter(*result, i);
    }
    c++;
  }
}

template <class List>
List *Random::sample(const List &population, ::size_t count) {
  List *result = new List();

  struct Inserter {
    void operator()(List &list, typename List::value_type &item) {
      list.insert(item);
    }
  };
  sample(result, population, count, Inserter());
#ifndef NDEBUG
  for (auto i : *result) {
    assert(population.find(i) != population.end());
  }
#endif
  return result;
}

template <class Element>
std::vector<Element> *Random::sample(const std::vector<Element> &population,
                                     ::size_t count) {
  std::unordered_set<int> accu;
  std::vector<Element> *result = new std::vector<Element>(accu.size());

  struct Inserter {
    void operator()(std::vector<Element> &list, Element &item) {
      list.push_back(item);
    }
  };
  sample(result, population, count, Inserter());

  return result;
}

template <class Container>
std::list<typename Container::value_type> *Random::sampleList(
    const Container &population, ::size_t count) {
  std::list<typename Container::value_type> *result =
      new std::list<typename Container::value_type>();
  struct Inserter {
    void operator()(std::list<typename Container::value_type> &list,
                    typename Container::value_type &item) {
      list.push_back(item);
    }
  };
  sample(result, population, count, Inserter());

#ifndef NDEBUG
  for (auto i : *result) {
    assert(population.find(i) != population.end());
  }
#endif

  return result;
}

}  // namespace Random
}  // namespace mcmc

#endif  // ndef MCMC_RANDOM_H__
