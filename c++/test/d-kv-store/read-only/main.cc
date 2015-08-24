
#include <chrono>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#endif
#include <boost/program_options.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>

#include <mcmc/random.h>
#include <mcmc/options.h>

#include <dkvstore/DKVStoreFile.h>
#ifdef MCMC_ENABLE_RAMCLOUD
#include <dkvstore/DKVStoreRamCloud.h>
#endif
#ifdef MCMC_ENABLE_RDMA
#include <infiniband/verbs.h>
#include <dkvstore/DKVStoreRDMA.h>
#endif

#include <mcmc/timer.h>

typedef std::chrono::high_resolution_clock hires;
typedef std::chrono::duration<double> duration;

namespace po = boost::program_options;

using mcmc::timer::Timer;

static double GB(::size_t n, ::size_t k) {
  return static_cast<double>(n * k * sizeof(double)) /
    (1 << 30);
}

template <typename T>
std::vector<const T*>& constify(std::vector<T*>& v) {
  // Compiler doesn't know how to automatically convert
  // std::vector<T*> to std::vector<T const*> because the way
  // the template system works means that in theory the two may
  // be specialised differently.  This is an explicit conversion.
  return reinterpret_cast<std::vector<const T*>&>(v);
}

template <typename DKVStore>
class DKVWrapper {
 public:
  DKVWrapper(const mcmc::Options &options,
             const std::vector<std::string> &remains)
      : options_(options), remains_(remains) {
    d_kv_store_.PurgeKVRecords();
  }

  void run() {

    Timer outer("Outer time");
    outer.start();

    int64_t seed;
    ::size_t N;   // #nodes in the graph

    bool no_populate;
    bool unidirectional;
    bool do_write;
    bool single_request;
    bool no_random_request;
    bool check_duplicates;
    bool verify;

    std::string dkv_type_string;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("network,N",
       po::value< ::size_t>(&N)->default_value(1 << 20),
       "nodes in the network")
      ("no-populate,P",
       po::bool_switch(&no_populate)->default_value(false),
       "do not populate at start of run")
      ("unidirectional,1",
       po::bool_switch(&unidirectional)->default_value(false),
       "unidirectional requests")
      ("write,W",
       po::bool_switch(&do_write)->default_value(false),
       "write (i.s.o. read)")
      ("single,S",
       po::bool_switch(&single_request)->default_value(false),
       "use neighbor keys i.s.o. random keys")
      ("no-random,R",
       po::bool_switch(&no_random_request)->default_value(false),
       "use neighbor keys i.s.o. random keys")
      ("seed",
       po::value<int64_t>(&seed)->default_value(42),
       "random seed")
      ("check-duplicates,d",
       po::bool_switch(&check_duplicates)->default_value(false),
       "check keys for duplicates")
      ("verify,V",
       po::bool_switch(&verify)->default_value(false),
       "verify values")
      ;

    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(remains_).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    try {
      po::notify(vm);
    } catch (po::error &e) {
      std::cerr << "Option error: " << e.what() << std::endl;
      exit(33);
    }

    if (options_.help) {
      std::cout << desc << std::endl;
      return;
    }

    std::vector<std::string> remains = po::collect_unrecognized(parsed.options,
                                                                po::include_positional);

    // bool no_populate = vm["no-populate"].as<bool>();
    bool bidirectional = ! unidirectional;
    bool random_request = ! no_random_request;
    // bool single_request = vm["single"].as<bool>();
    // bool do_write = vm["write"].as<bool>();
    bool do_read = ! do_write;      // later maybe finer control
    // bool check_duplicates = vm["check-duplicates"].as<bool>();
    // bool verify = vm["verify"].as<bool>();

    int32_t n_hosts = 1;        // default: sequential
    int32_t rank = 0;           // default: sequential
    const char *prun_pe_hosts = getenv("PRUN_PE_HOSTS");
    if (prun_pe_hosts != NULL) {
      std::string trimmed(prun_pe_hosts);
      boost::trim(trimmed);
      std::vector<std::string> host_list;
      boost::split(host_list, trimmed, boost::is_any_of(" "));
      n_hosts = static_cast<int32_t>(host_list.size());
    } else {
      prun_pe_hosts = getenv("OMPI_COMM_WORLD_SIZE");
      if (prun_pe_hosts == NULL) {
          std::cerr << "Cannot determine run size from environment, assume sequential" << std::endl;
      } else {
        try {
          n_hosts = boost::lexical_cast<int32_t>(prun_pe_hosts);
        } catch (boost::bad_lexical_cast const&) {
          std::cerr << "Cannot determine run size from environment, assume sequential" << std::endl;
        }
      }
    }

    const char *prun_cpu_rank = getenv("PRUN_CPU_RANK");
    if (prun_cpu_rank == NULL) {
      prun_cpu_rank = getenv("OMPI_COMM_WORLD_RANK");
    }
    if (prun_cpu_rank == NULL) {
      std::cerr << "Cannot determine run rank from environment, assume sequential" << std::endl;
    } else {
      try {
        rank = boost::lexical_cast<int32_t>(prun_cpu_rank);
      } catch (boost::bad_lexical_cast const&) {
        std::cerr << "Cannot determine run rank from environment, assume sequential" << std::endl;
      }
    }

    ::size_t K = options_.K;                            // #communities
    ::size_t m = options_.mini_batch_size;          // #nodes in minibatch, total
    ::size_t n = options_.num_node_sample;          // #neighbors for each minibatch node
    ::size_t iterations = options_.max_iteration;

    if (m == 0 || n == 0) {
      throw mcmc::InvalidArgumentException("m and n must both be > 0");
    }

    ::size_t my_m;
    if (d_kv_store_.include_master()) {
      my_m = (m + n_hosts - 1) / n_hosts;
    } else {
      my_m = (m + (n_hosts - 1) - 1) / (n_hosts - 1);
    }
    ::size_t average_m = (m + n_hosts - 1) / n_hosts;

    try {
      d_kv_store_.Init(K, N, my_m * n, my_m, remains);
    } catch (po::error &e) {
      std::cerr << "Option error: " << e.what() << std::endl;
      return;
    }

    d_kv_store_.barrier();

    mcmc::Random::Random random(seed + rank);

    std::cout << "N " << N << " K " << K <<
      " m " << m << " my_m " << my_m << " average_m " << average_m <<
      " n " << n <<
      " hosts " << n_hosts << " rank " << rank <<
      " seed " << seed << std::endl;
    std::cout << "bidirectional " << bidirectional << " random " << random_request << " single-request " << single_request << " read " << do_read << " write " << do_write << std::endl;

    auto t = hires::now();
    // populate
    if (! no_populate) {
      if (d_kv_store_.include_master() || rank != 0) {
        ::size_t from = rank;
        ::size_t to   = N;
        ::size_t chunk = n_hosts;
        if (! d_kv_store_.include_master()) {
          from--;
          chunk--;
        }
        std::cerr << "********* Populate with keys " << from << ".." << to << " step " << chunk << std::endl;
        for (::size_t i = from; i < to; i += chunk) {
          // std::vector<double> pi = random.randn(K);
          std::vector<double> pi(K);
          for (::size_t k = 0; k < K; k++) {
            pi[k] = i + (double)k / K;
          }
          std::vector<int32_t> k(1, static_cast<int32_t>(i));
          std::vector<const double *> v(1, pi.data());
          d_kv_store_.WriteKVRecords(k, v);
        }
      }
      duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
      std::cout << "Populate " << N << "x" << K << " takes " <<
        (1000.0 * dur.count()) << "ms thrp " << (GB(N, K) / dur.count()) <<
        " GB/s" << std::endl;

      d_kv_store_.barrier();
    }

    std::vector<double *> cache(average_m * n);
    for (::size_t iter = 0; iter < iterations; ++iter) {
      // Vector of requests
      std::cerr << "********* " << iter << ": Sample the neighbors" <<
        std::endl;
      std::vector<int32_t> *neighbor;
      if (single_request) {
          neighbor = new std::vector<int32_t>(average_m * n);
          for (::size_t i = 0; i < average_m * n; ++i) {
              (*neighbor)[i] = (rank + 1) % N;
          }
      } else if (random_request) {
          if (average_m * n * 2 >= N) {
            std::cerr << "Warning: sampling " << (average_m * n) << " from " << N << " might take a long time" << std::endl;
          }
          neighbor = random.sampleRange(N, average_m * n);
      } else {
          neighbor = new std::vector<int32_t>(average_m * n);
          for (::size_t i = 0; i < average_m * n; ++i) {
              (*neighbor)[i] = (i * n_hosts + rank + 1) % N;
          }
      }
     
      if (check_duplicates) {
        for (::size_t i = 0; i < neighbor->size() - 1; ++i) {
          if (std::find(neighbor->begin() + i + 1, neighbor->end(), (*neighbor)[i]) != neighbor->end()) {
            std::cerr << "neighbor sample[" << i << "] has duplicate value " << (*neighbor)[i] << std::endl;
          }
        }
      }

      if (do_read) {
        if (bidirectional || rank == 0) {
          std::cout << "*********" << iter << ":  Start reading KVs... " <<
            std::endl;
          // Read the values for the neighbors
          auto t = hires::now();
          d_kv_store_.ReadKVRecords(cache, *neighbor,
                                    DKV::RW_MODE::READ_ONLY);
          duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
          std::cout << average_m << " Read " << average_m << "x" << n << "x" << K <<
            " takes " << (1000.0 * dur.count()) << "ms thrp " <<
            (GB(average_m * n, K) / dur.count()) << " GB/s" << std::endl;
          if (false) {
            for (::size_t i = 0; i < average_m * n; ++i) {
              std::cerr << "Key " << (*neighbor)[i] << " pi = {";
              for (::size_t k = 0; k < K; k++) {
                std::cerr << cache[i][k] << " ";
              }
              std::cerr << "}" << std::endl;
            }
          }
          if (verify) {
            int miss = 0;
            for (::size_t i = 0; i < average_m * n; ++i) {
              int32_t key = (*neighbor)[i];
              for (::size_t k = 0; k < K; k++) {
                if (cache[i][k] != key + (double)k / K) {
                  miss++;
                  std::cerr << "Ooppss... key " << key << " wrong value[" << k << "] " << cache[i][k] << " should be " << (key + (double)k / K) << std::endl;
                }
              }
            }
            std::cout << "Verify: checked " << (average_m * n) << " vectors, wrong values: " << miss << std::endl;
          }
        }
      }

      if (do_write) {
        if (bidirectional || rank == 0) {
          std::cout << "*********" << iter << ":  Start reading KVs... " <<
            std::endl;
          // Read the values for the neighbors
          auto t = hires::now();
          d_kv_store_.WriteKVRecords(*neighbor, constify(cache));
          duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
          std::cout << average_m << " Write " << average_m << "x" << n << "x" << K <<
            " takes " << (1000.0 * dur.count()) << "ms thrp " <<
            (GB(average_m * n, K) / dur.count()) << " GB/s" << std::endl;
          if (false) {
            for (::size_t i = 0; i < average_m * n; ++i) {
              std::cerr << "Key " << (*neighbor)[i] << " pi = {";
              for (::size_t k = 0; k < K; k++) {
                std::cerr << cache[i][k] << " ";
              }
              std::cerr << "}" << std::endl;
            }
          }
        }
      }

      if (false) {
        std::cout << "*********" << iter << ":  Sync... " << std::endl;
        d_kv_store_.barrier();
      }

      d_kv_store_.PurgeKVRecords();

      std::cout << "*********" << iter << ":  Sync... " << std::endl;
      d_kv_store_.barrier();
      std::cerr << "*********" << iter << ":  Sync done" << std::endl;

      delete neighbor;
    }

    outer.stop();
    std::cout << outer << std::endl;
  }

protected:
  const mcmc::Options &options_;
  const std::vector<std::string> &remains_;
  DKVStore d_kv_store_;
};

int main(int argc, char *argv[]) {
  std::cout << "Pid " << getpid() << " invoked with options: ";
  for (int i = 0; i < argc; ++i) {
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;

    mcmc::Options options(argc, argv);

	DKV::TYPE::TYPE dkv_type;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("dkv.type",
       po::value<DKV::TYPE::TYPE>(&dkv_type)->multitoken()->default_value(DKV::TYPE::FILE),
       "D-KV store type (file/ramcloud/rdma)")
      ;

    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(options.getRemains()).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    // po::basic_command_line_parser<char> clp(options.getRemains());
    // clp.options(desc).allow_unregistered.run();
    // po::store(clp.run(), vm);
    po::notify(vm);

    if (options.help) {
        std::cout << desc << std::endl;
        return 0;
    }

	std::cerr << "D-KV store " << dkv_type << std::endl;

    std::vector<std::string> remains = po::collect_unrecognized(parsed.options, po::include_positional);
    std::cerr << "main has unparsed options: \"";
    for (auto r : remains) {
      std::cerr << r << " ";
    }
    std::cerr << "\"" << std::endl;

    switch (dkv_type) {
	case DKV::TYPE::FILE: {
        DKVWrapper<DKV::DKVFile::DKVStoreFile> dkv_store(options, remains);
        dkv_store.run();
        break;
    }
#ifdef MCMC_ENABLE_RAMCLOUD
	case DKV::TYPE::RAMCLOUD: {
        DKVWrapper<DKV::DKVRamCloud::DKVStoreRamCloud> dkv_store(options, remains);
        dkv_store.run();
        break;
    }
#endif
#ifdef MCMC_ENABLE_RDMA
	case DKV::TYPE::RDMA: {
        DKVWrapper<DKV::DKVRDMA::DKVStoreRDMA> dkv_store(options, remains);
        dkv_store.run();
        break;
    }
#endif
    }

    return 0;
}
