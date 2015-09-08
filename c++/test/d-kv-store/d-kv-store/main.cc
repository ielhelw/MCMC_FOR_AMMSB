
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
#include <mcmc/timer.h>

#include <dkvstore/DKVStoreFile.h>
#ifdef MCMC_ENABLE_RAMCLOUD
#include <dkvstore/DKVStoreRamCloud.h>
#endif
#ifdef MCMC_ENABLE_RDMA
#include <infiniband/verbs.h>
#include <dkvstore/DKVStoreRDMA.h>
#endif

typedef std::chrono::high_resolution_clock hires;
typedef std::chrono::duration<double> duration;

namespace po = boost::program_options;

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
  DKVWrapper(const mcmc::Options &options, const std::vector<std::string> &remains)
      : options_(options), remains_(remains) {
  }

  void run() {

    mcmc::timer::Timer outer("Outer time");
    outer.start();

    int64_t seed;
    ::size_t N;   // #nodes in the graph

    std::string dkv_type_string;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("help", "help")
      ("network,N",
       po::value< ::size_t>(&N)->default_value(1 << 20),
       "nodes in the network")
      ("no-populate,P",
       po::bool_switch()->default_value(false),
       "do not populate at start of run")
      ("verify,V",
       po::bool_switch()->default_value(false),
       "verify contents")
      ("seed,S",
       po::value<int64_t>(&seed)->default_value(42),
       "random seed")
      ;

    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(remains_).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    po::notify(vm);

    if (vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return;
    }

    std::vector<std::string> remains = po::collect_unrecognized(parsed.options,
                                                                po::include_positional);

    d_kv_store_ = std::unique_ptr<DKVStore>(new DKVStore(remains));
    bool no_populate = vm["no-populate"].as<bool>();
    bool verify = vm["verify"].as<bool>();

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

    ::size_t K = options_.K;                            // #communities
    ::size_t m = options_.mini_batch_size;          // #nodes in minibatch, total
    ::size_t n = options_.num_node_sample;          // #neighbors for each minibatch node
    ::size_t iterations = options_.max_iteration;

	if (m == 0 || n == 0) {
		throw mcmc::MCMCException("minibatch size and neighborset size must be > 0");
	}

    ::size_t my_m = (m + n_hosts - 1) / n_hosts;

    d_kv_store_->Init(K, N, my_m * n, my_m);

    mcmc::Random::Random random(seed + rank);

    std::cout << "N " << N << " K " << K <<
      " m " << m << " my_m " << my_m << " n " << n <<
      " hosts " << n_hosts << " rank " << rank <<
      " seed " << seed << std::endl;

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
          pi[k] = i + (double)k / K;
        }
        std::vector<int32_t> k(1, static_cast<int32_t>(i));
        std::vector<const double *> v(1, pi.data());
        d_kv_store_->WriteKVRecords(k, v);
        d_kv_store_->PurgeKVRecords();
      }
      duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
      std::cout << "Populate " << N << "x" << K << " takes " <<
        (1000.0 * dur.count()) << "ms thrp " << (GB(N, K) / dur.count()) <<
        " GB/s" << std::endl;
    }

    std::vector<double *> cache(my_m * n);
    std::vector<double *> overwrite = d_kv_store_->GetWriteKVRecords(my_m);
    for (::size_t iter = 0; iter < iterations; ++iter) {
      std::vector<int32_t> *minibatch = random.sampleRange(N, my_m);

      // Vector for the neighbors
      std::cerr << "********* " << iter << ": Sample the neighbors" <<
        std::endl;
      std::vector<int32_t> *neighbor;
      if (my_m * n * 2 >= N) {
        std::cerr << "Warning: sampling " << (my_m * n) << " from " << N << " might take a long time" << std::endl;
      }
      neighbor = random.sampleRange(N, my_m * n);

      {
        std::cout << "*********" << iter << ":  Start reading KVs... " <<
          std::endl;
        // Read the values for the neighbors
        auto t = hires::now();
        d_kv_store_->ReadKVRecords(cache, *neighbor,
                                  DKV::RW_MODE::READ_ONLY);
        duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
        std::cout << my_m << " Read " << my_m << "x" << n << "x" << K <<
          " takes " << (1000.0 * dur.count()) << "ms thrp " <<
          (GB(my_m * n, K) / dur.count()) << " GB/s" << std::endl;
        if (verify) {
          int miss = 0;
          for (::size_t i = 0; i < my_m * n; ++i) {
            int32_t key = (*neighbor)[i];
            for (::size_t k = 0; k < K; k++) {
              if (cache[i][k] < key + (double)k / K || cache[i][k] > key + (double)k / K + (double)(iter + 1) / (K * K)) {
                miss++;
                std::cerr << "Ooppss... key " << key << " wrong value[" << k << "] " << cache[i][k] << " should be (slightly over) " << (key + (double)k / K) << std::endl;
              }
            }
          }
          std::cout << "Verify: checked " << (my_m * n) << " vectors, wrong values: " << miss << std::endl;
        }
      }

      std::cout << "*********" << iter << ":  Sync... " << std::endl;

      {
        std::cout << "*********" << iter << ":  Start writing KVs... " <<
          std::endl;
        for (::size_t i = 0; i < minibatch->size(); i++) {
          for (::size_t k = 0; k < K; k++) {
            overwrite[i][k] = (*minibatch)[i] + (double)k / K + (double)(iter + 1) / (K * K);
          }
        }

        // Write new values to the KV-store
        auto t = hires::now();
        d_kv_store_->WriteKVRecords(*minibatch, constify(overwrite));
        duration dur = std::chrono::duration_cast<duration>(hires::now() - t);
        std::cout << my_m << " Write " << my_m << "x" << K << " takes " <<
          (1000.0 * dur.count()) << " thrp " << (GB(my_m, K) / dur.count()) <<
          " GB/s" << std::endl;
      }

      d_kv_store_->PurgeKVRecords();

      delete neighbor;

      std::cout << "*********" << iter << ":  Sync... " << std::endl;

      delete minibatch;
    }

    outer.stop();
    std::cout << outer << std::endl;
  }

protected:
  const mcmc::Options &options_;
  const std::vector<std::string> &remains_;
  std::unique_ptr<DKVStore> d_kv_store_;
};

int main(int argc, char *argv[]) {
    mcmc::Options options(argc, argv);

    DKV::TYPE dkv_type;
    po::options_description desc("D-KV store test program");
    desc.add_options()
      ("help", "help")
      ("dkv.type",
       po::value<DKV::TYPE>(&dkv_type)->multitoken()->default_value(DKV::TYPE::FILE),
       "D-KV store type (file/ramcloud/rdma)")
      ;

    po::variables_map vm;
    po::parsed_options parsed = po::basic_command_line_parser<char>(options.getRemains()).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    po::notify(vm);

    if (vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::vector<std::string> remains = po::collect_unrecognized(parsed.options, po::include_positional);
    std::cerr << "main has unparsed options: \"";
    for (auto r : remains) {
      std::cerr << r << " ";
    }
    std::cerr << "\"" << std::endl;

    switch (dkv_type) {
	case DKV::TYPE::FILE: {
#if 0
        DKVWrapper<DKV::DKVFile::DKVStoreFile> dkv_store(options, remains);
        dkv_store.run();
#endif
        break;
    }
#ifdef MCMC_ENABLE_RAMCLOUD
	case DKV::TYPE::RAMCLOUD: {
#if 0
        DKVWrapper<DKV::DKVRamCloud::DKVStoreRamCloud> dkv_store(options, remains);
        dkv_store.run();
#endif
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
