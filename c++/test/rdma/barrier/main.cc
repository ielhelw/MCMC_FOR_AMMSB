
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

#include <mcmc/random.h>
#include <mcmc/options.h>

#include <d-kv-store/file/DKVStoreFile.h>
#ifdef ENABLE_RAMCLOUD
#include <d-kv-store/ramcloud/DKVStoreRamCloud.h>
#endif
#ifdef ENABLE_RDMA
#include <infiniband/verbs.h>
#include <d-kv-store/rdma/DKVStoreRDMA.h>
#endif

#include <mcmc/timer.h>
#include <mr/timer.h>

typedef std::chrono::high_resolution_clock hires;
typedef std::chrono::duration<double> duration;

namespace po = boost::program_options;

using mr::timer::Timer;

static double GB(::size_t n, ::size_t k) {
  return static_cast<double>(n * k * sizeof(double)) /
    (1 << 30);
}

namespace DKV_TYPE {
enum TYPE {
    FILE,
#ifdef ENABLE_RAMCLOUD
    RAMCLOUD,
#endif
#ifdef ENABLE_RDMA
    RDMA,
#endif
};
}   // namespace DKV_TYPE


namespace DKV {
namespace DKVRDMA {
extern struct ibv_device **global_dev_list;
}
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
    CHECK_DEV_LIST();
    d_kv_store_.Info();
    CHECK_DEV_LIST();

    d_kv_store_.InfoH();
    CHECK_DEV_LIST();

    d_kv_store_.PurgeKVRecords();
    CHECK_DEV_LIST();
  }

  void run() {

    Timer outer("Outer time");
    outer.start();

    int64_t seed;
    ::size_t N;   // #nodes in the graph
    CHECK_DEV_LIST();

    std::string dkv_type_string;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("network,N",
       po::value< ::size_t>(&N)->default_value(1 << 20),
       "nodes in the network")
      ("no-populate,P",
       po::bool_switch()->default_value(false),
       "do not populate at start of run")
      ("single-source,1",
       po::bool_switch()->default_value(false),
       "single source of requests")
      ("write,W",
       po::bool_switch()->default_value(false),
       "write (i.s.o. read)")
      ("no-random,R",
       po::bool_switch()->default_value(false),
       "use neighbor keys i.s.o. random keys")
      ("seed,S",
       po::value<int64_t>(&seed)->default_value(42),
       "random seed")
      ("check-duplicates,d",
       po::bool_switch()->default_value(false),
       "check keys for duplicates")
      ;

    CHECK_DEV_LIST();
    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(remains_).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    try {
      po::notify(vm);
      CHECK_DEV_LIST();
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

    CHECK_DEV_LIST();
    bool no_populate = vm["no-populate"].as<bool>();
    bool single_source = vm["single-source"].as<bool>();
    bool random_request = ! vm["no-random"].as<bool>();
    bool do_write = vm["write"].as<bool>();
    bool do_read = ! do_write;      // later maybe finer control
    bool check_duplicates = vm["check-duplicates"].as<bool>();

    int32_t n_hosts;
    int32_t rank;
    const char *prun_pe_hosts = getenv("NHOSTS");
    if (prun_pe_hosts == NULL) {
      prun_pe_hosts = getenv("OMPI_COMM_WORLD_SIZE");
    }
    const char *prun_cpu_rank = getenv("PRUN_CPU_RANK");
    if (prun_cpu_rank == NULL) {
      prun_cpu_rank = getenv("OMPI_COMM_WORLD_RANK");
    }

    try {
      n_hosts = boost::lexical_cast<int32_t>(prun_pe_hosts);
      rank    = boost::lexical_cast<int32_t>(prun_cpu_rank);
    } catch (boost::bad_lexical_cast const&) {
      std::cerr << "Cannot determine run size/rank from environment, assume sequential" << std::endl;
      n_hosts = 1;
      rank    = 0;
    }

    CHECK_DEV_LIST();

    ::size_t K = options_.K;                            // #communities
    ::size_t m = options_.mini_batch_size;          // #nodes in minibatch, total
    ::size_t n = options_.num_node_sample;          // #neighbors for each minibatch node
    ::size_t iterations = options_.max_iteration;

    ::size_t my_m = (m + n_hosts - 1) / n_hosts;

CHECK_DEV_LIST();
d_kv_store_.Info();
CHECK_DEV_LIST();
d_kv_store_.PurgeKVRecords();
CHECK_DEV_LIST();

    try {
      d_kv_store_.Init(K, N, my_m * n, my_m, remains);
    } catch (po::error &e) {
      std::cerr << "Option error: " << e.what() << std::endl;
      return;
    }

    d_kv_store_.barrier();

    mcmc::Random::Random random(seed + rank);

    std::cout << "N " << N << " K " << K <<
      " m " << m << " my_m " << my_m << " n " << n <<
      " hosts " << n_hosts << " rank " << rank <<
      " seed " << seed << std::endl;
    std::cout << "single-source " << single_source << " random " << random_request << " read " << do_read << " write " << do_write << std::endl;

    auto t = hires::now();
    // populate
    if (! no_populate) {
      ::size_t from = rank;
      ::size_t to   = N;
      std::cerr << "********* Populate with keys " << from << ".." << to << " step " << n_hosts << std::endl;
      for (::size_t i = from; i < to; i += n_hosts) {
        // std::vector<double> pi = random.randn(K);
        std::vector<double> pi(K);
        for (::size_t k = 0; k < K; k++) {
          pi[k] = i * 1000.0 + (k + 1) / 1000.0;
        }
        std::vector<int32_t> k(1, static_cast<int32_t>(i));
        std::vector<const double *> v(1, pi.data());
        d_kv_store_.WriteKVRecords(k, v);
      }
      duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
      std::cout << "Populate " << N << "x" << K << " takes " <<
        (1000.0 * dur.count()) << "ms thrp " << (GB(N, K) / dur.count()) <<
        " GB/s" << std::endl;
    }

    std::vector<double *> cache(my_m * n);
    for (::size_t iter = 0; iter < iterations; ++iter) {
      // Vector of requests
      std::cerr << "********* " << iter << ": Sample the neighbors" <<
        std::endl;
      std::vector<int32_t> *neighbor;
      if (random_request) {
          if (my_m * n * 2 >= N) {
            std::cerr << "Warning: sampling " << (my_m * n) << " from " << N << " might take a long time" << std::endl;
          }
          neighbor = random.sampleRange(N, my_m * n);
      } else {
          neighbor = new std::vector<int32_t>(my_m * n);
          for (::size_t i = 0; i < my_m * n; ++i) {
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
        if (! single_source || rank == 0) {
          std::cout << "*********" << iter << ":  Start reading KVs... " <<
            std::endl;
          // Read the values for the neighbors
          auto t = hires::now();
          d_kv_store_.ReadKVRecords(cache, *neighbor,
                                    DKV::RW_MODE::READ_ONLY);
          duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
          std::cout << my_m << " Read " << my_m << "x" << n << "x" << K <<
            " takes " << (1000.0 * dur.count()) << "ms thrp " <<
            (GB(my_m * n, K) / dur.count()) << " GB/s" << std::endl;
          if (false) {
            for (::size_t i = 0; i < my_m * n; ++i) {
              std::cerr << "Key " << (*neighbor)[i] << " pi = {";
              for (::size_t k = 0; k < K; k++) {
                std::cerr << cache[i][k] << " ";
              }
              std::cerr << "}" << std::endl;
            }
          }
        }
      }

      if (do_write) {
        if (! single_source || rank == 0) {
          std::cout << "*********" << iter << ":  Start reading KVs... " <<
            std::endl;
          // Read the values for the neighbors
          auto t = hires::now();
          d_kv_store_.WriteKVRecords(*neighbor, constify(cache));
          duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
          std::cout << my_m << " Write " << my_m << "x" << n << "x" << K <<
            " takes " << (1000.0 * dur.count()) << "ms thrp " <<
            (GB(my_m * n, K) / dur.count()) << " GB/s" << std::endl;
          if (false) {
            for (::size_t i = 0; i < my_m * n; ++i) {
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

  int num_devices;
  DKV::DKVRDMA::global_dev_list = ibv_get_device_list(&num_devices);
  std::cerr << "IB devices: " << num_devices << std::endl;
  for (int i = 0; i < num_devices; i++) {
    std::cerr << "  IB device[" << i << "] device_name " << (void *)DKV::DKVRDMA::global_dev_list[0] << " " << DKV::DKVRDMA::global_dev_list[i]->dev_name << std::endl;
  }

  std::cout << "Invoked with options: ";
  for (int i = 0; i < argc; ++i) {
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;

    mcmc::Options options(argc, argv);

    CHECK_DEV_LIST();

    std::string dkv_type_string;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("dkv.type",
       po::value<std::string>(&dkv_type_string)->default_value("file"),
       "D-KV store type (file/ramcloud/rdma)")
      ;

    CHECK_DEV_LIST();
    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(options.getRemains()).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    // po::basic_command_line_parser<char> clp(options.getRemains());
    // clp.options(desc).allow_unregistered.run();
    // po::store(clp.run(), vm);
    po::notify(vm);
    CHECK_DEV_LIST();

    if (options.help) {
        std::cout << desc << std::endl;
        return 0;
    }

    CHECK_DEV_LIST();

    DKV_TYPE::TYPE dkv_type = DKV_TYPE::FILE;
    if (vm.count("dkv.type") > 0) {
        if (false) {
        } else if (dkv_type_string == "file") {
            dkv_type = DKV_TYPE::FILE;
#ifdef ENABLE_RAMCLOUD
        } else if (dkv_type_string == "ramcloud") {
            dkv_type = DKV_TYPE::RAMCLOUD;
#endif
#ifdef ENABLE_RDMA
        } else if (dkv_type_string == "rdma") {
            dkv_type = DKV_TYPE::RDMA;
#endif
        } else {
            desc.print(std::cerr);
            throw mcmc::InvalidArgumentException("Unsupported value '" + dkv_type_string + "' for dkv.type");
        }
    }
    CHECK_DEV_LIST();

    std::vector<std::string> remains = po::collect_unrecognized(parsed.options, po::include_positional);
    std::cerr << "main has unparsed options: \"";
    for (auto r : remains) {
      std::cerr << r << " ";
    }
    std::cerr << "\"" << std::endl;

    switch (dkv_type) {
    case DKV_TYPE::FILE: {
#if 0
        DKVWrapper<DKV::DKVFile::DKVStoreFile> dkv_store(options, remains);
        dkv_store.run();
#endif
        break;
    }
#ifdef ENABLE_RAMCLOUD
    case DKV_TYPE::RAMCLOUD: {
#if 0
        DKVWrapper<DKV::DKVRamCloud::DKVStoreRamCloud> dkv_store(options, remains);
        dkv_store.run();
#endif
        break;
    }
#endif
#ifdef ENABLE_RDMA
    case DKV_TYPE::RDMA: {
        DKVWrapper<DKV::DKVRDMA::DKVStoreRDMA> dkv_store(options, remains);
        dkv_store.run();
        break;
    }
#endif
    }

    return 0;
}
