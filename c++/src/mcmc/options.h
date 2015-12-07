#ifndef MCMC_OPTIONS_H__
#define MCMC_OPTIONS_H__

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "mcmc/config.h"
#include "mcmc/types.h"
#include "mcmc/exception.h"

#include "dkvstore/DKVStore.h"

namespace mcmc {

/**
 * For want of a better place...
 */
template <typename T>
std::string to_string(T value) {
  std::ostringstream s;
  s << value;
  return s.str();
}

/**
 * utility function to parse a size_t.
 * @throw NumberFormatException if arg is malformed or out of range
 */

::size_t parse_size_t(const std::string &argString);

/**
 * utility function to parse an integral.
 * @throw NumberFormatException if arg is malformed or out of range
 */

template <class T>
inline T parse(const std::string &argString) {
  const char *arg = argString.c_str();

  if (arg[0] == '-') {
    ::ssize_t result = -parse_size_t(arg + 1);
    if (result < (::ssize_t)std::numeric_limits<T>::min()) {
      throw mcmc::NumberFormatException("number too small for type");
    }
    return (T)result;
  } else {
    ::size_t result = parse_size_t(arg);
    if (result > (::size_t)std::numeric_limits<T>::max()) {
      throw mcmc::NumberFormatException("number too large for type");
    }
    return (T)result;
  }
}

template <>
inline float parse<float>(const std::string &argString) {
  float f;

  if (sscanf(argString.c_str(), "%f", &f) != 1) {
    throw mcmc::NumberFormatException("string is not a float");
  }

  return f;
}

template <class T>
class KMG {
 public:
  KMG() {}

  KMG(const std::string &s) { f = mcmc::parse<T>(s); }

  T operator()() { return f; }

 protected:
  T f;
};

template <class T>
inline std::istream &operator>>(std::istream &in, KMG<T> &n) {
  std::string line;
  std::getline(in, line);
  n = KMG<T>(line);
  return in;
}

}  // namespace mcmc

#include <boost/program_options.hpp>

// Specializations for boost validate (parse) calls for the most common
// integer types, so we can use shorthand like 32m for 32*(2<<20)
//
namespace boost {
namespace program_options {

#define VALIDATE_SPECIALIZE(T)                                             \
  template <>                                                              \
  inline void validate<T, char>(                                           \
      boost::any & v, const std::vector<std::basic_string<char> > &xs, T *, \
      long) {                                                              \
    validators::check_first_occurrence(v);                                 \
    std::basic_string<char> s(validators::get_single_string(xs));          \
    try {                                                                  \
      T x = mcmc::parse<T>(s);                                             \
      v = any(x);                                                          \
    } catch (mcmc::NumberFormatException e) {                              \
      (void) e;                                                            \
      boost::throw_exception(invalid_option_value(s));                     \
    }                                                                      \
  }

// VALIDATE_SPECIALIZE(::size_t)
// VALIDATE_SPECIALIZE(::ssize_t)
VALIDATE_SPECIALIZE(int32_t)
VALIDATE_SPECIALIZE(uint32_t)
VALIDATE_SPECIALIZE(int64_t)
VALIDATE_SPECIALIZE(uint64_t)
}
}

namespace mcmc {

namespace po = ::boost::program_options;

class Options {
 public:
  Options()
    : desc_all("Options"),
      desc_mcmc("MCMC Options"),
      desc_io("MCMC Input Options")
#ifdef MCMC_ENABLE_DISTRIBUTED
      , desc_distr("MCMC Distributed Options")
#endif
  {
    desc_all.add_options()
      ("config", po::value<std::string>(&config_file), "config file")
    ;
    desc_mcmc.add_options()
      // mcmc options
      ("mcmc.alpha", po::value<Float>(&alpha)->default_value(0.0), "alpha")
      ("mcmc.eta0", po::value<Float>(&eta0)->default_value(1.0), "eta0")
      ("mcmc.eta1", po::value<Float>(&eta1)->default_value(1.0), "eta1")
      ("mcmc.epsilon",
       po::value<Float>(&epsilon)->default_value(0.0000001), "epsilon")
      ("mcmc.a", po::value<Float>(&a)->default_value(0.0), "a")
      ("mcmc.b", po::value<Float>(&b)->default_value(1024), "b")
      ("mcmc.c", po::value<Float>(&c)->default_value(0.5), "c")
      ("mcmc.K,K", po::value< ::size_t>(&K)->default_value(300), "K")
      ("mcmc.mini-batch-size,m",
       po::value< ::size_t>(&mini_batch_size)->default_value(0),
       "mini_batch_size")
      ("mcmc.num-node-sample,n",
       po::value< ::size_t>(&num_node_sample)->default_value(0),
       "neighbor sample size")
      ("mcmc.strategy",
       po::value<strategy::strategy>(&strategy)->multitoken()->default_value(
          strategy::STRATIFIED_RANDOM_NODE),
       "sampling strategy")
      ("mcmc.max-iteration,x",
       po::value< ::size_t>(&max_iteration)->default_value(10000000),
       "max_iteration")
      ("mcmc.interval,i",
       po::value< ::size_t>(&interval)->default_value(0),
       "perplexity interval")
      ("mcmc.num-updates",
       po::value< ::size_t>(&num_updates)->default_value(1000),
       "num_updates")
      ("mcmc.held-out-ratio,h",
       po::value<double>(&held_out_ratio)->default_value(0.0),
       "held_out_ratio")
      ("mcmc.seed",
       po::value<int>(&random_seed)->default_value(42), "random seed")
      ;
    desc_all.add(desc_mcmc);

      // input options 
    desc_io.add_options()
      ("mcmc.input.file,f",
       po::value<std::string>(&input_filename_)->default_value(""),
       "input file")
      ("mcmc.input.class,c",
       po::value<std::string>(&input_class_)->default_value("relativity"),
       "input class")
      ("mcmc.input.contiguous",
       po::bool_switch(&input_contiguous_)->default_value(false),
       "contiguous input data")
      ("mcmc.input.compressed",
       po::bool_switch(&input_compressed_)->default_value(false),
       "compressed input data")
    ;
    desc_all.add(desc_io);
#ifdef MCMC_ENABLE_DISTRIBUTED
    desc_distr.add_options()
      ("mcmc.dkv-type",
       po::value<DKV::TYPE>(&dkv_type)->multitoken()->default_value(
#ifdef MCMC_ENABLE_RDMA
         DKV::TYPE::RDMA
#elif defined MCMC_ENABLE_RAMCLOUD
         DKV::TYPE::RAMCLOUD
#else
         DKV::TYPE::FILE
#endif
         ),
       "D-KV store type (file/ramcloud/rdma)")
      ("mcmc.num-buffers",
       po::value< ::size_t>(&num_buffers_)->default_value(2),
       "#buffers for update-phi multibuffering")
      ("mcmc.max-pi-cache",
       po::value< ::size_t>(&max_pi_cache_entries_)->default_value(0),
       "minibatch chunk size")
      ("mcmc.master_is_worker",
       po::bool_switch(&forced_master_is_worker)->default_value(false),
       "master host also is a worker")
      ("mcmc.replicated-graph",
       po::bool_switch(&REPLICATED_NETWORK)->default_value(false),
       "replicate Network graph")
    ;
    desc_all.add(desc_distr);
#endif
  }

  Options(int argc, char** argv) : Options() {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
    Parse(args);
  }
  
  Options(const std::vector<std::string>& args) : Options() {
    Parse(args);
  }

  void Parse(const std::vector<std::string>& argv) {
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argv)
                                    .options(desc_all)
                                    .allow_unregistered()
                                    .run();
    po::store(parsed, vm);
    po::notify(vm);
    if (vm.count("config") > 0) {
      std::ifstream file(config_file);
      po::store(po::parse_config_file(file, desc_all), vm);
      po::notify(vm);
    }
    remains = po::collect_unrecognized(parsed.options, po::include_positional);

    if (a == 0.0) {
      a = std::pow(b, -c);
    }
    if (alpha == 0.0) {
      alpha = 1.0 / K;
    }
  }

  const std::vector<std::string> &getRemains() const { return remains; }

 public:
    
  std::string config_file;

  Float alpha;
  Float eta0;
  Float eta1;
  ::size_t K;
  ::size_t mini_batch_size;
  ::size_t num_node_sample;
  strategy::strategy strategy;
  Float epsilon;
  ::size_t max_iteration;
  ::size_t interval;

  // parameters for step size
  Float a;
  Float b;
  Float c;

  ::size_t num_updates;
  double held_out_ratio;

  std::string input_filename_;
  std::string input_class_;
  bool input_contiguous_;
  bool input_compressed_;

  int random_seed;

  std::vector<std::string> remains;
#ifdef MCMC_ENABLE_DISTRIBUTED
  DKV::TYPE dkv_type;
  bool forced_master_is_worker;
  mutable ::size_t	max_pi_cache_entries_;
  ::size_t num_buffers_;
  bool REPLICATED_NETWORK;
#endif
  po::options_description desc_all;
  po::options_description desc_mcmc;
  po::options_description desc_io;
#ifdef MCMC_ENABLE_DISTRIBUTED
  po::options_description desc_distr;
#endif

  friend std::ostream& operator<<(std::ostream& out, const Options& opts);
};

inline std::ostream& operator<<(std::ostream& out, const Options& opts) {
  out << opts.desc_all;
  return out;
}

};  // namespace mcmc

#endif  // ndef MCMC_OPTIONS_H__
