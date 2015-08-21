#ifndef MCMC_OPTIONS_H__
#define MCMC_OPTIONS_H__

#include <iostream>
#include <fstream>

#include "mcmc/exception.h"

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#endif
#include <boost/program_options.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif

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
      boost::any & v, const std::vector<std::basic_string<char>> &xs, T *, \
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
  Options(int argc, char *argv[],
          const po::options_description *caller_options = NULL) {
    std::string config_file;

    po::options_description desc("Options");

    desc.add_options()("help,?", "help")(
        "config", po::value<std::string>(&config_file), "config file")

        ("run.stochastical,s", "MCMC Stochastical C++")("run.batch,t",
                                                        "MCMC Batch C++")
#ifdef ENABLE_OPENCL
        ("run.stochastical-opencl,S", "MCMC Stochastical OpenCL")(
            "run.batch-opencl,T", "MCMC Batch OpenCL")
#endif
        ("run.stochastical-distributed,D", "MCMC Stochastical Distributed")

        ("mcmc.alpha", po::value<double>(&alpha)->default_value(0.01), "alpha")(
            "mcmc.eta0", po::value<double>(&eta0)->default_value(1.0), "eta0")(
            "mcmc.eta1", po::value<double>(&eta1)->default_value(1.0), "eta1")(
            "mcmc.epsilon,e",
            po::value<double>(&epsilon)->default_value(0.0000001), "epsilon")(
            "mcmc.a", po::value<double>(&a)->default_value(0.01), "a")(
            "mcmc.b", po::value<double>(&b)->default_value(1024), "b")(
            "mcmc.c", po::value<double>(&c)->default_value(0.55), "c")

        ("mcmc.K,K", po::value<::size_t>(&K)->default_value(300), "K")(
            "mcmc.mini-batch-size,m",
            po::value<::size_t>(&mini_batch_size)->default_value(0),
            "mini_batch_size")(
            "mcmc.num-node-sample,n",
            po::value<::size_t>(&num_node_sample)->default_value(0),
            "neighbor sample size")(
            "mcmc.strategy",
            po::value<std::string>(&strategy)->default_value("unspecified"),
            "sampling strategy")

        ("mcmc.max-iteration,x",
         po::value<::size_t>(&max_iteration)->default_value(10000000),
         "max_iteration")("mcmc.interval,i",
                          po::value<::size_t>(&interval)->default_value(0),
                          "perplexity interval")

        ("mcmc.num-updates,u",
         po::value<::size_t>(&num_updates)->default_value(1000),
         "num_updates")("mcmc.held-out-ratio,h",
                        po::value<double>(&held_out_ratio)->default_value(0.0),
                        "held_out_ratio")

        ("output.dir,o",
         po::value<std::string>(&output_dir)->default_value("."), "output_dir")

        ("input.file,f", po::value<std::string>(&filename)->default_value(""),
         "input file")(
            "input.class,c",
            po::value<std::string>(&dataset_class)->default_value("netscience"),
            "input class")("input.contiguous,C",
                           po::bool_switch(&contiguous)->default_value(false),
                           "contiguous input data")(
            "input.compressed,z",
            po::bool_switch(&compressed)->default_value(false),
            "compressed input data")

#ifdef ENABLE_OPENCL
        ("cl.platform,p", po::value<std::string>(&openClPlatform),
         "OpenCL platform")("cl.device,d",
                            po::value<std::string>(&openClDevice),
                            "OpenCL device")(
            "cl.thread-group-size,G",
            po::value<::size_t>(&openclGroupSize)->default_value(1),
            "OpenCL thread group size")(
            "cl.num-thread-groups,g",
            po::value<::size_t>(&openclNumGroups)->default_value(1),
            "num OpenCL thread groups")(
            "cl.buffer-size,b",
            po::value<::size_t>(&openclBufferSize)->default_value(0),
            "OpenCL buffer size")
#endif
        ;

    if (caller_options != NULL) {
      desc.add(*caller_options);
    }

    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv)
                                    .options(desc)
                                    .allow_unregistered()
                                    .run();
    po::store(parsed, vm);
    po::notify(vm);
    if (vm.count("config") > 0) {
      std::ifstream file(config_file);
      po::store(po::parse_config_file(file, desc), vm);
      po::notify(vm);
    }

    remains = po::collect_unrecognized(parsed.options, po::include_positional);

    help = vm.count("help") > 0;
    if (help) {
      std::cout << desc << std::endl;
    }

    run.mcmc_stochastical = vm.count("run.stochastical") > 0;
    run.mcmc_batch = vm.count("run.batch") > 0;
#ifdef ENABLE_OPENCL
    run.mcmc_stochastical_cl = vm.count("run.stochastical-opencl") > 0;
    run.mcmc_batch_cl = vm.count("run.batch-opencl") > 0;
#endif
    run.mcmc_stochastical_distr = vm.count("run.stochastical-distributed") > 0;
  }

  const std::vector<std::string> &getRemains() const { return remains; }

 public:
  bool help;

  double alpha;
  double eta0;
  double eta1;
  ::size_t K;
  ::size_t mini_batch_size;
  ::size_t num_node_sample;
  std::string strategy;
  double epsilon;
  ::size_t max_iteration;
  ::size_t interval;

  // parameters for step size
  double a;
  double b;
  double c;

  ::size_t num_updates;
  double held_out_ratio;
  std::string output_dir;

  std::string filename;
  std::string dataset_class;
  bool contiguous;
  bool compressed;

#ifdef ENABLE_OPENCL
  std::string openClPlatform;
  std::string openClDevice;
  ::size_t openclGroupSize;
  ::size_t openclNumGroups;
  ::size_t openclBufferSize;
#endif

  struct {
    bool mcmc_stochastical;
    bool mcmc_batch;
#ifdef ENABLE_OPENCL
    bool mcmc_stochastical_cl;
    bool mcmc_batch_cl;
#endif
    bool mcmc_stochastical_distr;
  } run;

  std::vector<std::string> remains;
};

};  // namespace mcmc

#endif  // ndef MCMC_OPTIONS_H__
