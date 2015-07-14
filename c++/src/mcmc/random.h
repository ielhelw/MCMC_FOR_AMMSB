#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "mcmc/exception.h"

namespace mcmc {
namespace Random {

#ifdef RANDOM_FOLLOWS_CPP_WENZHE
#define RANDOM_SYSTEM
#endif

// #define RANDOM_SYSTEM
// #define USE_TAUS2_RANDOM

class Random {
 protected:
  Random();

 public:
  Random(unsigned int seed);

  Random(unsigned int seed_hi, unsigned int seed_lo);

  virtual ~Random();

#ifndef RANDOM_SYSTEM
  inline uint64_t xorshift_128plus();

  uint64_t seed(int x) const;

  inline uint64_t rand();
#else
  const uint64_t seed(int x);
#endif

  int64_t randint(int64_t from, int64_t upto);

  double random();

  std::vector<std::vector<double> > randn(::size_t K, ::size_t N);

  template <class List>
  List *sample(const List &population, ::size_t count);

  template <class List>
  List *sample(const List *population, ::size_t count);

  template <class Element>
  std::vector<Element> *sample(const std::vector<Element> &population,
                               ::size_t count);

  std::vector<int> *sampleRange(int N, ::size_t count);

  template <class Container>
  std::list<typename Container::value_type> *sampleList(
      const Container &population, ::size_t count);

  template <class Container>
  std::list<typename Container::value_type> *sampleList(
      const Container *population, ::size_t count);

  std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1,
                                          ::size_t n2);

  std::vector<double> randn(::size_t K);

  void report();

  std::string state();

 protected:
  std::unordered_set<int> sample(int from, int upto, ::size_t count);

  template <class Input, class Result, class Inserter>
  void sample(Result *result, const Input &input, ::size_t count,
              Inserter inserter);

#ifndef RANDOM_SYSTEM
 protected:
  double gsl_ran_gaussian_ziggurat(const double sigma);

  double gsl_rng_uniform_pos();

  double gsl_ran_gamma(double a, double b);

  uint64_t xorshift_state[2];

  ::size_t n_randn = 0;
  ::size_t iters_randn = 0;
  ::size_t log_exp_randn = 0;
  ::size_t ktab_exceed_randn = 0;

#else
#if __GNUC_MINOR__ >= 5
  std::default_random_engine generator;
  std::normal_distribution<double> normalDistribution;
#else  // if __GNUC_MINOR__ >= 5
  throw UnimplementedException("random::gamma");
#endif
#endif  // ndef RANDOM_SYSTEM
};

class FileReaderRandom : public Random {
 public:
  FileReaderRandom(unsigned int seed);

  virtual ~FileReaderRandom();

  std::vector<double> randn(::size_t K);

  std::vector<std::vector<double> > randn(::size_t K, ::size_t N);

  double random();

  int64_t randint(int64_t from, int64_t upto);

  template <class List>
  List *sample(const List &population, ::size_t count);

  template <class List>
  List *sample(const List *population, ::size_t count);

  template <class Element>
  std::vector<Element> *sample(const std::vector<Element> &population,
                               ::size_t count);

  std::vector<int> *sampleRange(int N, ::size_t count);

  template <class Container>
  std::list<typename Container::value_type> *sampleList(
      const Container &population, ::size_t count);

  template <class Container>
  std::list<typename Container::value_type> *sampleList(
      const Container *population, ::size_t count);

  std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1,
                                          ::size_t n2);

 protected:
  void getline(std::ifstream &f, std::string &line);

  std::ifstream floatReader;
  std::ifstream intReader;
  std::ifstream sampleReader;
  std::ifstream choiceReader;
  std::ifstream gammaReader;
  std::ifstream noiseReader;
};

#ifdef RANDOM_FOLLOWS_PYTHON
extern FileReaderRandom *random;
#else
extern Random *random;
#endif

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

template <class List>
List *Random::sample(const List *population, ::size_t count) {
  return sample(*population, count);
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

template <class Container>
std::list<typename Container::value_type> *Random::sampleList(
    const Container *population, ::size_t count) {
  return sampleList(*population, count);
}

// FileReaderRandom
template <class List>
List *FileReaderRandom::sample(const List &population, ::size_t count) {
  std::string line;
  List *result = new List();
  getline(sampleReader, line);

  std::istringstream is(line);

  for (::size_t i = 0; i < count; i++) {
    typename List::key_type key(is);
    result->insert(key);
  }
  return result;
}
template <class List>
List *FileReaderRandom::sample(const List *population, ::size_t count) {
  return sample(*population, count);
}
template <class Element>
std::vector<Element> *FileReaderRandom::sample(
    const std::vector<Element> &population, ::size_t count) {
  std::string line;
  getline(sampleReader, line);
  std::istringstream is(line);

  std::vector<Element> *result = new std::vector<Element>(count);

  for (::size_t i = 0; i < count; i++) {
    int r;

    if (!(is >> r)) {
      throw IOException("end of line");
    }
    result->push_back(r);
  }
  return result;
}

template <class Container>
std::list<typename Container::value_type> *FileReaderRandom::sampleList(
    const Container &population, ::size_t count) {
  std::string line;
  auto *result = new std::list<typename Container::value_type>();
  getline(sampleReader, line);

  std::istringstream is(line);

  for (::size_t i = 0; i < count; i++) {
    typename Container::value_type key(is);
    result->push_back(key);
  }
  return result;
}

template <class Container>
std::list<typename Container::value_type> *FileReaderRandom::sampleList(
    const Container *population, ::size_t count) {
  return sampleList(*population, count);
}

}  // namespace Random
}  // namespace mcmc

#endif  // ndef MCMC_RANDOM_H__
