#include <mcmc/learning/mcmc_sampler_stochastic_distr.h>

#include <cinttypes>
#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "mcmc/exception.h"

#ifdef MCMC_ENABLE_DISTRIBUTED
#  include <mpi.h>
#else

#define MPI_SUCCESS		0

typedef int MPI_Comm;
#define MPI_COMM_WORLD	0

enum MPI_ERRORS {
  MPI_ERRORS_RETURN,
  MPI_ERRORS_ARE_FATAL,
};

enum MPI_Datatype {
  MPI_INT                    = 0x4c000405,
  MPI_LONG                   = 0x4c000407,
  MPI_UNSIGNED_LONG          = 0x4c000408,
  MPI_DOUBLE                 = 0x4c00080b,
  MPI_BYTE                   = 0x4c00010d,
};

enum MPI_Op {
  MPI_SUM,
};

void *MPI_IN_PLACE = (void *)0x88888888;


int MPI_Init(int *argc, char ***argv) {
  return MPI_SUCCESS;
}

int MPI_Finalize() {
  return MPI_SUCCESS;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
  return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm) {
  return MPI_SUCCESS;
}

int MPI_Comm_set_errhandler(MPI_Comm comm, int mode) {
  return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int *mpi_size) {
  *mpi_size = 1;
  return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int *mpi_rank) {
  *mpi_rank = 0;
  return MPI_SUCCESS;
}

::size_t mpi_datatype_size(MPI_Datatype type) {
  switch (type) {
  case MPI_INT:
    return sizeof(int32_t);
  case MPI_LONG:
    return sizeof(int64_t);
  case MPI_UNSIGNED_LONG:
    return sizeof(uint64_t);
  case MPI_DOUBLE:
    return sizeof(double);
  case MPI_BYTE:
    return 1;
  default:
    std::cerr << "Unknown MPI datatype" << std::cerr;
    return 0;
  }
}

int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm) {
  memcpy(recvbuf, sendbuf, sendcount * mpi_datatype_size(sendtype));
  return MPI_SUCCESS;
}

int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
  return MPI_Scatter((char *)sendbuf + displs[0] * mpi_datatype_size(sendtype),
                     sendcounts[0], sendtype,
                     recvbuf, recvcount, recvtype,
                     root, comm);
}

int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) {
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * mpi_datatype_size(datatype));
  }
  return MPI_SUCCESS;
}

#endif

#include "dkvstore/DKVStoreFile.h"
#ifdef MCMC_ENABLE_RAMCLOUD
#include "dkvstore/DKVStoreRamCloud.h"
#endif
#ifdef MCMC_ENABLE_RDMA
#include "dkvstore/DKVStoreRDMA.h"
#endif

#include "mcmc/np.h"


namespace mcmc {
namespace learning {

#define PRINT_MEM_USAGE() \
do { \
  std::cerr << __func__ << "():" << __LINE__ << " "; \
  print_mem_usage(std::cerr); \
} while (0)

typedef OrderedVertexSet NeighborSet;

using ::mcmc::timer::Timer;


// **************************************************************************
//
// class LocalNetwork
//
// **************************************************************************

void LocalNetwork::unmarshall_local_graph(::size_t index, const Vertex* linked,
                                          ::size_t size) {
  if (linked_edges_.size() <= index) {
    linked_edges_.resize(index + 1);
  }
  linked_edges_[index] = EndpointSet();
  for (::size_t i = 0; i < size; ++i) {
    linked_edges_[index].insert(linked[i]);
  }
}

void LocalNetwork::reset() {
  linked_edges_.clear();
}

bool LocalNetwork::find(const Edge& edge) const {
  const auto &adj = linked_edges_[edge.first];

  return adj.find(edge.second) != adj.end();
}

const LocalNetwork::EndpointSet& LocalNetwork::linked_edges(::size_t i) const {
  return linked_edges_[i];
}


// **************************************************************************
//
// class PerpData
//
// **************************************************************************

void PerpData::Init(::size_t max_perplexity_chunk) {
  // Convert the vertices into their rank, that is all we need

  // Find the ranks
  nodes_.resize(data_.size() * 2);
  Vertex ix = 0;
  for (auto edge : data_) {
    const Edge &e = edge.edge;
    nodes_[ix] = e.first;
    ++ix;
    nodes_[ix] = e.second;
    ++ix;
  }

  pi_.resize(2 * max_perplexity_chunk);

  accu_.resize(omp_get_max_threads());
}



// **************************************************************************
//
// class MCMCSamplerStochasticDistributed
//
// **************************************************************************
MCMCSamplerStochasticDistributed::MCMCSamplerStochasticDistributed(
    const Options &args) : MCMCSamplerStochastic(args), mpi_master_(0) {
  t_populate_pi_           = Timer("  populate pi");
  t_outer_                 = Timer("  iteration");
  t_deploy_minibatch_      = Timer("    deploy minibatch");
  t_mini_batch_            = Timer("      sample_mini_batch");
  t_nodes_in_mini_batch_   = Timer("      nodes_in_mini_batch");
  t_sample_neighbor_nodes_ = Timer("      sample_neighbor_nodes");
  t_load_pi_minibatch_     = Timer("      load minibatch pi");
  t_load_pi_neighbor_      = Timer("      load neighbor pi");
  t_update_phi_            = Timer("      update_phi");
  t_update_phi_in_         = Timer("      update_phi in graph");
  t_barrier_phi_           = Timer("    barrier to update phi");
  t_update_pi_             = Timer("    update_pi");
  t_store_pi_minibatch_    = Timer("      store minibatch pi");
  t_barrier_pi_            = Timer("    barrier to update pi");
  t_update_beta_           = Timer("    update_beta");
  t_beta_zero_             = Timer("      zero beta grads");
  t_beta_rank_             = Timer("      rank minibatch nodes");
  t_beta_calc_grads_       = Timer("      beta calc grads");
  t_beta_sum_grads_        = Timer("      beta sum grads");
  t_beta_update_theta_     = Timer("      update theta");
  t_load_pi_beta_          = Timer("      load pi update_beta");
  t_broadcast_beta_        = Timer("      broadcast beta");
  t_perplexity_            = Timer("  perplexity");
  t_load_pi_perp_          = Timer("      load perplexity pi");
  t_cal_edge_likelihood_   = Timer("      calc edge likelihood");
  t_purge_pi_perp_         = Timer("      purge perplexity pi");
  t_reduce_perp_           = Timer("      reduce/plus perplexity");
  Timer::setTabular(true);
}


MCMCSamplerStochasticDistributed::~MCMCSamplerStochasticDistributed() {
  delete d_kv_store_;

  for (auto &p : pi_update_) {
    delete[] p;
  }

  delete phi_init_rng_;
  for (auto r : neighbor_sample_rng_) {
    delete r;
  }
  for (auto r : phi_update_rng_) {
    delete r;
  }

  (void)MPI_Finalize();
}


void MCMCSamplerStochasticDistributed::BroadcastNetworkInfo() {
  NetworkInfo info;
  int r;

  if (mpi_rank_ == mpi_master_) {
    network.FillInfo(&info);
  }

  r = MPI_Bcast(&info, sizeof info, MPI_BYTE, mpi_master_, MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Bcast of Network stub info fails");

  if (mpi_rank_ != mpi_master_) {
    network = Network(info);

    N = network.get_num_nodes();
    assert(N != 0);

    beta = std::vector<double>(K, 0.0);
    pi   = std::vector<std::vector<double> >(N, std::vector<double>(K, 0.0));

    // parameters related to sampling
    mini_batch_size = args_.mini_batch_size;
    if (mini_batch_size < 1) {
      mini_batch_size = N / 2;    // default option.
    }

    // ration between link edges and non-link edges
    link_ratio = network.get_num_linked_edges() / ((N * (N - 1)) / 2.0);

    ppx_for_heldout = std::vector<double>(network.get_held_out_size(), 0.0);

    this->info(std::cerr);
  }
}


void MCMCSamplerStochasticDistributed::BroadcastHeldOut() {
  int r;
  int32_t my_held_out_size;

  if (mpi_rank_ == mpi_master_) {
    std::vector<int32_t> count(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> displ(mpi_size_);	// FIXME: lift to class

    if (REPLICATED_NETWORK) {
      // Ensure perplexity is centrally calculated at the master's
      for (int i = 0; i < mpi_size_; ++i) {
        if (i == mpi_master_) {
          count[i] = network.get_held_out_set().size();
        } else {
          count[i] = 0;
        }
      }
    } else {
      int32_t held_out_marshall_size = network.get_held_out_set().size() /
                                         mpi_size_;
      ::size_t surplus = network.get_held_out_set().size() % mpi_size_;
      for (::size_t i = 0; i < surplus; ++i) {
        count[i] = held_out_marshall_size + 1;
      }
      for (::size_t i = surplus; i < static_cast<::size_t>(mpi_size_); ++i) {
        count[i] = held_out_marshall_size;
      }
    }

    // Scatter the size of each held-out set subset
    r = MPI_Scatter(count.data(), 1, MPI_INT,
                    &my_held_out_size, 1, MPI_INT,
                    mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatter of held_out_set size fails");

    // Marshall the subsets
    std::vector<EdgeMapItem> buffer(network.get_held_out_set().size());
    struct EdgeMapItem* p = buffer.data();

    for (auto e : network.get_held_out_set()) {
      p->edge = e.first;
      p->is_edge = e.second;
      ++p;
    }

    std::vector<int32_t> bytes(mpi_size_);
    for (::size_t i = 0; i < count.size(); ++i) {
      bytes[i] = count[i] * sizeof(EdgeMapItem);
    }
    displ[0] = 0;
    for (int i = 1; i < mpi_size_; ++i) {
      displ[i] = displ[i - 1] + bytes[i];
    }
    // Scatter the marshalled subgraphs
    perp_.data_.resize(my_held_out_size);
    r = MPI_Scatterv(buffer.data(), bytes.data(), displ.data(), MPI_BYTE,
                     perp_.data_.data(),
                     perp_.data_.size() * sizeof(EdgeMapItem), MPI_BYTE,
                     mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of held-out set data fails");

  } else {
    // Scatter the fanout of each minibatch node
    r = MPI_Scatter(NULL, 1, MPI_INT,
                    &my_held_out_size, 1, MPI_INT,
                    mpi_master_,
                    MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatter of held_out_set size fails");

    // Scatter the marshalled subgraphs
    perp_.data_.resize(my_held_out_size);
    r = MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE,
                     perp_.data_.data(),
                     perp_.data_.size() * sizeof(EdgeMapItem), MPI_BYTE,
                     mpi_master_, MPI_COMM_WORLD);
  }
}


void MCMCSamplerStochasticDistributed::MasterAwareLoadNetwork() {
  if (REPLICATED_NETWORK) {
    LoadNetwork(mpi_rank_);
  } else {
    if (mpi_rank_ == mpi_master_) {
      LoadNetwork(mpi_rank_);
    }
    BroadcastNetworkInfo();
    // No need to broadcast the Network aux stuff, fan_out_cumul_distro and
    // cumulative_edges: it is used at the master only
  }
  BroadcastHeldOut();
}


void MCMCSamplerStochasticDistributed::init() {
  int r;

  // In an OpenMP program: no need for thread support
  r = MPI_Init(NULL, NULL);
  mpi_error_test(r, "MPI_Init() fails");

  r = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  mpi_error_test(r, "MPI_Comm_set_errhandler fails");

  r = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
  mpi_error_test(r, "MPI_Comm_size() fails");
  r = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  mpi_error_test(r, "MPI_Comm_rank() fails");

  DKV::TYPE::TYPE dkv_type;
  bool forced_master_is_worker;

  po::options_description desc("MCMC Stochastic Distributed");

  max_pi_cache_entries_ = 0;
  desc.add_options()
    ("dkv.type",
     po::value<DKV::TYPE::TYPE>(&dkv_type)->multitoken()->default_value(DKV::TYPE::TYPE::FILE),
     "D-KV store type (file/ramcloud/rdma)")
    ("mcmc.max-pi-cache",
     po::value<::size_t>(&max_pi_cache_entries_)->default_value(0),
     "minibatch chunk size")
    ("mcmc.master_is_worker",
     po::bool_switch(&forced_master_is_worker)->default_value(false),
     "master host also is a worker")
    ("mcmc.replicated-graph",
     po::bool_switch(&REPLICATED_NETWORK)->default_value(false),
     "replicate Network graph")
    ;

  po::variables_map vm;
  po::parsed_options parsed =
    po::basic_command_line_parser<char>(
      args_.getRemains()).options(desc).allow_unregistered().run();
  po::store(parsed, vm);
  // po::basic_command_line_parser<char> clp(options.getRemains());
  // clp.options(desc).allow_unregistered.run();
  // po::store(clp.run(), vm);
  po::notify(vm);

  std::vector<std::string> dkv_args =
    po::collect_unrecognized(parsed.options, po::include_positional);

  std::cerr << "Use D-KV store type " << dkv_type << std::endl;
  switch (dkv_type) {
  case DKV::TYPE::FILE:
    d_kv_store_ = new DKV::DKVFile::DKVStoreFile();
    break;
#ifdef MCMC_ENABLE_RAMCLOUD
  case DKV::TYPE::RAMCLOUD:
    d_kv_store_ = new DKV::DKVRamCloud::DKVStoreRamCloud();
    break;
#endif
#ifdef MCMC_ENABLE_RDMA
  case DKV::TYPE::RDMA:
    d_kv_store_ = new DKV::DKVRDMA::DKVStoreRDMA();
    break;
#endif
  }

  if (forced_master_is_worker) {
    master_is_worker_ = true;
  } else {
    master_is_worker_ = (mpi_size_ == 1);
  }

  MasterAwareLoadNetwork();

  // control parameters for learning
  //num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
  if (args_.num_node_sample == 0) {
    // TODO: automative update..... 
    num_node_sample = N/50;
  } else {
    num_node_sample = args_.num_node_sample;
  }
  if (args_.mini_batch_size == 0) {
    // old default for STRATIFIED_RANDOM_NODE_SAMPLING
    mini_batch_size = N / 10;
  }
  std::cerr << "num_node_sample " << num_node_sample <<
    " a " << a << " b " << b << " c " << c << " alpha " << alpha <<
    " eta (" << eta[0] << "," << eta[1] << ")" << std::endl;

  if (max_pi_cache_entries_ == 0) {
    std::ifstream meminfo("/proc/meminfo");
    int64_t mem_total = -1;
    while (meminfo.good()) {
      char buffer[256];
      char* colon;

      meminfo.getline(buffer, sizeof buffer);
      if (strncmp("MemTotal", buffer, 8) == 0 &&
           (colon = strchr(buffer, ':')) != 0) {
        if (sscanf(colon + 2, "%" SCNd64, &mem_total) != 1) {
          throw NumberFormatException("MemTotal must be a longlong");
        }
        break;
      }
    }
    if (mem_total == -1) {
      throw InvalidArgumentException(
              "/proc/meminfo has no line for MemTotal");
    }
    // /proc/meminfo reports KB
    ::size_t pi_total = (1024 * mem_total) / ((K + 1) * sizeof(double));
    max_pi_cache_entries_ = pi_total / 32;
  }

  max_minibatch_nodes_ = network.minibatch_nodes_for_strategy(
                            mini_batch_size, strategy);
  ::size_t workers;
  if (master_is_worker_) {
    workers = mpi_size_;
  } else {
    workers = mpi_size_ - 1;
  }
  // pi cache hosts chunked subset of minibatch nodes + their neighbors
  max_minibatch_chunk_ = max_pi_cache_entries_ / (1 + real_num_node_sample());
  ::size_t max_my_minibatch_nodes = std::min(max_minibatch_chunk_,
                                             (max_minibatch_nodes_ +
                                                workers - 1) /
                                             workers);
  ::size_t max_minibatch_neighbors = max_my_minibatch_nodes *
                                      real_num_node_sample();

  // for perplexity, cache pi for both vertexes of each edge
  max_perplexity_chunk_ = max_pi_cache_entries_ / 2;
  ::size_t num_perp_nodes = (2 * network.get_held_out_size() +
                             workers - 1) / workers;
  ::size_t max_my_perp_nodes = std::min(max_perplexity_chunk_,
                                        num_perp_nodes);

  if (mpi_rank_ == mpi_master_) {
    // master must cache pi[minibatch] for update_beta
    max_minibatch_neighbors = std::max(max_minibatch_neighbors,
                                       max_minibatch_nodes_);
    if (max_minibatch_neighbors > max_pi_cache_entries_) {
      throw MCMCException("pi cache cannot contain pi[minibatch] for beta, "
                          "refactor so update_beta is chunked");
    }
  }

  ::size_t max_pi_cache = std::max(max_my_minibatch_nodes +
                                     max_minibatch_neighbors,
                                   max_my_perp_nodes);

  std::cerr << "minibatch size param " << mini_batch_size <<
    " max " << max_minibatch_nodes_ <<
    " chunk " << max_minibatch_chunk_ <<
    " #neighbors(total) " << max_minibatch_neighbors << std::endl;
  std::cerr << "perplexity nodes total " <<
    (network.get_held_out_size() * 2) <<
    " local " << num_perp_nodes <<
    " mine " << max_my_perp_nodes <<
    " chunk " << max_perplexity_chunk_ << std::endl;

  d_kv_store_->Init(K + 1, N, max_pi_cache, max_my_minibatch_nodes, dkv_args);

  master_hosts_pi_ = d_kv_store_->include_master();

  std::cerr << "Master is " << (master_is_worker_ ? "" : "not ") <<
    "a worker, does " << (master_hosts_pi_ ? "" : "not ") <<
    "host pi values" << std::endl;

  // Need to know max_perplexity_chunk_ to Init perp_
  perp_.Init(max_perplexity_chunk_);

  if (mpi_rank_ == mpi_master_) {
    init_beta();
  }

  // Make phi_update_rng_ depend on mpi_rank_ and thread Id
  phi_update_rng_.resize(omp_get_max_threads());
  int seed;
  seed = args_.random_seed + SourceAwareRandom::PHI_UPDATE;
  for (::size_t i = 0; i < phi_update_rng_.size(); ++i) {
    int my_seed = seed + 1 + i + mpi_rank_ * phi_update_rng_.size();
    phi_update_rng_[i] = new Random::Random(my_seed, seed,
                                            RANDOM_PRESERVE_RANGE_ORDER);
  }

  // Make neighbor_sample_rng_ depend on mpi_rank_ and thread Id
  neighbor_sample_rng_.resize(omp_get_max_threads());
  seed = args_.random_seed + SourceAwareRandom::NEIGHBOR_SAMPLER;
  for (::size_t i = 0; i < phi_update_rng_.size(); ++i) {
    int my_seed = seed + 1 + i + mpi_rank_ * phi_update_rng_.size();
    neighbor_sample_rng_[i] = new Random::Random(my_seed, seed,
                                                 RANDOM_PRESERVE_RANGE_ORDER);
  }

  // Make phi_init_rng_ depend on mpi_rank_
  seed = args_.random_seed + SourceAwareRandom::PHI_INIT;
  phi_init_rng_ = new Random::Random(seed + 1 + mpi_rank_, seed,
                                     RANDOM_PRESERVE_RANGE_ORDER);

  t_populate_pi_.start();
  init_pi();
  t_populate_pi_.stop();
#ifdef DOESNT_WORK_FOR_DISTRIBUTED
  std::cout << "phi[0][0] " << phi[0][0] << std::endl;

  if (true) {
    for (::size_t i = 0; i < 10; ++i) {
      std::cerr << "phi[" << i << "]: ";
      for (::size_t k = 0; k < 10; ++k) {
        std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
      }
      std::cerr << std::endl;
      std::cerr << "pi[" << i << "]: ";
      for (::size_t k = 0; k < 10; ++k) {
        std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
      }
      std::cerr << std::endl;
    }
  }
#endif

  pi_update_.resize(max_minibatch_nodes_);
  for (auto &p : pi_update_) {
    p = new double[K + 1];
  }
  grads_beta_.resize(omp_get_max_threads());
  for (auto &g : grads_beta_) {
    g = std::vector<std::vector<double> >(K, std::vector<double>(2));    // gradients K*2 dimension
  }

  std::cerr << "Random seed " << std::hex << "0x" <<
    rng_.random(SourceAwareRandom::GRAPH_INIT)->seed(0) <<
    ",0x" <<
    rng_.random(SourceAwareRandom::GRAPH_INIT)->seed(1) <<
    std::endl;
  std::cerr << std::dec;
}


void MCMCSamplerStochasticDistributed::run() {
  /** run mini-batch based MCMC sampler, based on the sungjin's note */

  PRINT_MEM_USAGE();

  using namespace std::chrono;

  std::vector<std::vector<double>> phi_node(max_minibatch_nodes_,
                                            std::vector<double>(K + 1));

  int r;

  r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Barrier(initial) fails");

  t_start_ = clock();
  while (step_count < max_iteration && ! is_converged()) {

    t_outer_.start();
    // auto l1 = std::chrono::system_clock::now();
    //if (step_count > 200000){
    //interval = 2;
    //}
    check_perplexity();

    t_broadcast_beta_.start();
    r = MPI_Bcast(beta.data(), beta.size(), MPI_DOUBLE, mpi_master_,
                  MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Bcast of beta fails");
    t_broadcast_beta_.stop();

    t_deploy_minibatch_.start();
    // edgeSample is nonempty only at the master
    // assigns nodes_
    EdgeSample edgeSample = deploy_mini_batch();
    t_deploy_minibatch_.stop();
    std::cerr << "Minibatch nodes " << nodes_.size() << std::endl;

    update_phi(&phi_node);

    // all synchronize with barrier: ensure we read pi/phi_sum from current
    // iteration
    t_barrier_phi_.start();
    r = MPI_Barrier(MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Barrier(post phi) fails");
    t_barrier_phi_.stop();

    // TODO calculate and store updated values for pi/phi_sum
    t_update_pi_.start();
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < nodes_.size(); ++i) {
      pi_from_phi(pi_update_[i], phi_node[i]);
    }
    t_update_pi_.stop();
    // std::cerr << "write back phi/pi after update" << std::endl;
    t_store_pi_minibatch_.start();
    d_kv_store_->WriteKVRecords(nodes_, constify(pi_update_));
    t_store_pi_minibatch_.stop();
    d_kv_store_->PurgeKVRecords();

    // all synchronize with barrier
    t_barrier_pi_.start();
    r = MPI_Barrier(MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Barrier(post pi) fails");
    t_barrier_pi_.stop();

    if (mpi_rank_ == mpi_master_) {
      // TODO load pi/phi values for the minibatch nodes
      t_update_beta_.start();
      update_beta(*edgeSample.first, edgeSample.second);
      t_update_beta_.stop();

      // TODO FIXME allocate this outside the loop
      delete edgeSample.first;
    }

    ++step_count;
    t_outer_.stop();
    // auto l2 = std::chrono::system_clock::now();
  }

  check_perplexity();

  Timer::printHeader(std::cout);
  std::cout << t_populate_pi_ << std::endl;
  std::cout << t_outer_ << std::endl;
  std::cout << t_deploy_minibatch_ << std::endl;
  std::cout << t_mini_batch_ << std::endl;
  std::cout << t_nodes_in_mini_batch_ << std::endl;
  std::cout << "    update_phi" << std::endl;
  std::cout << t_sample_neighbor_nodes_ << std::endl;
  std::cout << t_load_pi_minibatch_ << std::endl;
  std::cout << t_load_pi_neighbor_ << std::endl;
  std::cout << t_update_phi_ << std::endl;
  std::cout << t_update_phi_in_ << std::endl;
  std::cout << t_barrier_phi_ << std::endl;
  std::cout << t_update_pi_ << std::endl;
  std::cout << t_store_pi_minibatch_ << std::endl;
  std::cout << t_purge_pi_perp_ << std::endl;
  std::cout << t_reduce_perp_ << std::endl;
  std::cout << t_barrier_pi_ << std::endl;
  std::cout << t_update_beta_ << std::endl;
  std::cout << t_beta_zero_ << std::endl;
  std::cout << t_beta_rank_ << std::endl;
  std::cout << t_load_pi_beta_ << std::endl;
  std::cout << t_beta_calc_grads_ << std::endl;
  std::cout << t_beta_sum_grads_ << std::endl;
  std::cout << t_beta_update_theta_ << std::endl;
  std::cout << t_broadcast_beta_ << std::endl;
  std::cout << t_perplexity_ << std::endl;
  std::cout << t_load_pi_perp_ << std::endl;
  std::cout << t_cal_edge_likelihood_ << std::endl;
  std::cout << t_reduce_perp_ << std::endl;
}


::size_t MCMCSamplerStochasticDistributed::real_num_node_sample() const {
  return num_node_sample + 1;
}


void MCMCSamplerStochasticDistributed::init_beta() {
  // TODO only at the Master
  // model parameters and re-parameterization
  // since the model parameter - \pi and \beta should stay in the simplex,
  // we need to restrict the sum of probability equals to 1.  The way we
  // restrict this is using re-reparameterization techniques, where we
  // introduce another set of variables, and update them first followed by
  // updating \pi and \beta.
  // parameterization for \beta
  theta = rng_.random(SourceAwareRandom::THETA_INIT)->gamma(eta[0], eta[1],
                                                            K, 2);
  // std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
  // parameterization for \beta
  // theta = rng_.random(SourceAwareRandom::THETA_INIT)->->gamma(100.0, 0.01, K, 2);

  std::vector<std::vector<double> > temp(theta.size(),
                                         std::vector<double>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<double>(1));

  if (true) {
    std::cout << std::fixed << std::setprecision(12) << "beta[0] " << beta[0] <<
      std::endl;
  } else {
    std::cerr << "beta ";
    for (::size_t k = 0; k < K; ++k) {
      std::cerr << std::fixed << std::setprecision(12) << beta[k] << " ";
    }
    std::cerr << std::endl;
  }

}


// Calculate pi[0..K> ++ phi_sum from phi[0..K>
void MCMCSamplerStochasticDistributed::pi_from_phi(
    double* pi, const std::vector<double> &phi) {
  double phi_sum = std::accumulate(phi.begin(), phi.begin() + K, 0.0);
  for (::size_t k = 0; k < K; ++k) {
    pi[k] = phi[k] / phi_sum;
  }

  pi[K] = phi_sum;
}


void MCMCSamplerStochasticDistributed::init_pi() {
  double pi[K + 1];
  std::cerr << "*************** FIXME: load pi only on pi-hoster nodes" <<
    std::endl;
  for (int32_t i = mpi_rank_; i < static_cast<int32_t>(N); i += mpi_size_) {
    std::vector<double> phi_pi = phi_init_rng_->gamma(1, 1, 1, K)[0];
    if (true) {
      if (i < 10) {
        std::cerr << "phi[" << i << "]: ";
        for (::size_t k = 0; k < std::min(K, 10UL); ++k) {
          std::cerr << std::fixed << std::setprecision(12) << phi_pi[k] << " ";
        }
        std::cerr << std::endl;
      }
    }
#ifndef NDEBUG
    for (auto ph : phi_pi) {
      assert(ph >= 0.0);
    }
#endif

    pi_from_phi(pi, phi_pi);
    if (true) {
      if (i < 10) {
        std::cerr << "pi[" << i << "]: ";
        for (::size_t k = 0; k < std::min(K, 10UL); ++k) {
          std::cerr << std::fixed << std::setprecision(12) << pi[k] << " ";
        }
        std::cerr << std::endl;
      }
    }

    std::vector<int32_t> node(1, i);
    std::vector<const double*> pi_wrapper(1, pi);
    // Easy performance improvement: accumulate records up to write area
    // size, the Write/Purge
    d_kv_store_->WriteKVRecords(node, pi_wrapper);
    d_kv_store_->PurgeKVRecords();
  }
}


void MCMCSamplerStochasticDistributed::check_perplexity() {
  if (step_count % interval == 0) {
    t_perplexity_.start();
    // TODO load pi for the held-out set to calculate perplexity
    double ppx_score = cal_perplexity_held_out();
    t_perplexity_.stop();
    std::cout << std::fixed << std::setprecision(12) <<
      "step count: " << step_count <<
      " perplexity for hold out set: " << ppx_score << std::endl;
    ppxs_held_out.push_back(ppx_score);

    clock_t t2 = clock();
    double diff = (double)t2 - (double)t_start_;
    double seconds = diff / CLOCKS_PER_SEC;
    timings_.push_back(seconds);
    iterations.push_back(step_count);
#if 0
    if (ppx_score < 5.0) {
      stepsize_switch = true;
      //print "switching to smaller step size mode!"
    }
#endif
  }

  // write into file
  if (step_count % 2000 == 1) {
    if (false) {
      throw MCMCException(
          "Implement parallel dump of perplexity[node] history");
      std::ofstream myfile;
      std::string file_name = "mcmc_stochastic_" + to_string(K) +
        "_num_nodes_" + to_string(num_node_sample) + "_us_air.txt";
      myfile.open (file_name);
      int size = ppxs_held_out.size();
      for (int i = 0; i < size; ++i){

        //int iteration = i * 100 + 1;
        myfile << iterations[i] << "    " << timings_[i] << "    " <<
          ppxs_held_out[i] << "\n";
      }

      myfile.close();
    }
  }
}


void MCMCSamplerStochasticDistributed::ScatterSubGraph(
    const std::vector<std::vector<int32_t>> &subminibatch) {
  std::vector<int32_t> set_size(nodes_.size());
  std::vector<Vertex> flat_subgraph;
  int r;

  local_network_.reset();

  if (mpi_rank_ == mpi_master_) {
    std::vector<int32_t> size_count(mpi_size_);	        // FIXME: lift to class
    std::vector<int32_t> size_displ(mpi_size_);	        // FIXME: lift to class
    std::vector<int32_t> subgraph_count(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> subgraph_displ(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> workers_set_size;

    // Data dependency on workers_set_size
    for (int i = 0; i < mpi_size_; ++i) {
      subgraph_count[i] = 0;
      for (::size_t j = 0; j < subminibatch[i].size(); ++j) {
        int32_t fan_out = network.get_fan_out(subminibatch[i][j]);
        workers_set_size.push_back(fan_out);
        subgraph_count[i] += fan_out;
      }
      size_count[i] = subminibatch[i].size();
    }

    size_displ[0] = 0;
    subgraph_displ[0] = 0;
    for (int i = 1; i < mpi_size_; ++i) {
      size_displ[i] = size_displ[i - 1] + size_count[i - 1];
      subgraph_displ[i] = subgraph_displ[i - 1] + subgraph_count[i - 1];
    }

    // Scatter the fanout of each minibatch node
    r = MPI_Scatterv(workers_set_size.data(),
                     size_count.data(),
                     size_displ.data(),
                     MPI_INT,
                     set_size.data(),
                     set_size.size(),
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");

    // Marshall the subgraphs
    ::size_t total_edges = np::sum(workers_set_size);
    std::vector<Vertex> subgraphs(total_edges);
    ::size_t marshalled = 0;
    for (int i = 0; i < mpi_size_; ++i) {
      for (::size_t j = 0; j < subminibatch[i].size(); ++j) {
        assert(marshalled + network.get_fan_out(subminibatch[i][j]) <=
               total_edges);
        Vertex* marshall = subgraphs.data() + marshalled;
        ::size_t n = network.marshall_edges_from(subminibatch[i][j],
                                                 marshall);
        // std::cerr << "Marshall to peer " << i << ": " << n <<
        //   " edges" << std::endl;
        marshalled += n;
      }
    }
    std::cerr << "Total marshalled " << marshalled <<
      " presumed " << total_edges << std::endl;
    assert(marshalled == total_edges);

    // Scatter the marshalled subgraphs
    ::size_t total_set_size = np::sum(set_size);
    flat_subgraph.resize(total_set_size);
    r = MPI_Scatterv(subgraphs.data(),
                     subgraph_count.data(),
                     subgraph_displ.data(),
                     MPI_INT,
                     flat_subgraph.data(),
                     flat_subgraph.size(),
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);

  } else {
    // Scatter the fanout of each minibatch node
    r = MPI_Scatterv(NULL,
                     NULL,
                     NULL,
                     MPI_INT,
                     set_size.data(),
                     set_size.size(),
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");

    // Scatter the marshalled subgraphs
    ::size_t total_set_size = np::sum(set_size);
    flat_subgraph.resize(total_set_size);
    r = MPI_Scatterv(NULL,
                     NULL,
                     NULL,
                     MPI_INT,
                     flat_subgraph.data(),
                     flat_subgraph.size(),
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);
  }

  ::size_t offset = 0;
  for (::size_t i = 0; i < set_size.size(); ++i) {
    Vertex* marshall = &flat_subgraph[offset];
    local_network_.unmarshall_local_graph(i, marshall, set_size[i]);
    offset += set_size[i];
    if (false) {
      std::cerr << "Vertex " << nodes_[i] << ": ";
      for (auto p : local_network_.linked_edges(i)) {
        std::cerr << p << " ";
      }
      std::cerr << std::endl;
      std::cerr << "Unmarshalled " << set_size[i] << " edges" << std::endl;
    }
  }
}


EdgeSample MCMCSamplerStochasticDistributed::deploy_mini_batch() {
  std::vector<std::vector<int>> subminibatch;
  std::vector<int32_t> minibatch_chunk(mpi_size_);      // FIXME: lift to class
  std::vector<int32_t> scatter_minibatch;               // FIXME: lift to class
  std::vector<int32_t> scatter_displs(mpi_size_);       // FIXME: lift to class
  int		r;
  EdgeSample edgeSample;

  if (mpi_rank_ == mpi_master_) {
    // std::cerr << "Invoke sample_mini_batch" << std::endl;
    t_mini_batch_.start();
    edgeSample = network.sample_mini_batch(mini_batch_size,
                                           strategy::STRATIFIED_RANDOM_NODE);
    t_mini_batch_.stop();
    const MinibatchSet &mini_batch = *edgeSample.first;
    if (true) {
      std::cerr << "Minibatch[" << mini_batch.size() << "]: ";
      for (auto e : mini_batch) {
        std::cerr << e << " ";
      }
      std::cerr << std::endl;
    }
    // std::cerr << "Done sample_mini_batch" << std::endl;

    // iterate through each node in the mini batch.
    t_nodes_in_mini_batch_.start();
    OrderedVertexSet nodes = nodes_in_batch(mini_batch);
    t_nodes_in_mini_batch_.stop();
    // std::cerr << "mini_batch size " << mini_batch.size() <<
    //   " num_node_sample " << num_node_sample << std::endl;

    subminibatch.resize(mpi_size_);	// FIXME: lift to class, size is static

    ::size_t workers = master_is_worker_ ? mpi_size_ : mpi_size_ - 1;
    ::size_t upper_bound = (nodes.size() + workers - 1) / workers;
    std::unordered_set<Vertex> unassigned;
    for (auto n: nodes) {
      ::size_t owner = node_owner(n);
      if (subminibatch[owner].size() == upper_bound) {
        unassigned.insert(n);
      } else {
        subminibatch[owner].push_back(n);
      }
    }
    std::cerr << "#nodes " << nodes.size() <<
      " #unassigned " << unassigned.size() <<
      " upb " << upper_bound << std::endl;

    std::cerr << "subminibatch[" << subminibatch.size() << "] = [";
    for (auto s : subminibatch) {
      std::cerr << s.size() << " ";
    }
    std::cerr << "]" << std::endl;

    ::size_t i = master_is_worker_ ? 0 : 1;
    for (auto n: unassigned) {
      while (subminibatch[i].size() == upper_bound) {
        ++i;
        assert(i < static_cast< ::size_t>(mpi_size_));
      }
      subminibatch[i].push_back(n);
    }

    scatter_minibatch.clear();
    int32_t running_sum = 0;
    for (int i = 0; i < mpi_size_; ++i) {
      minibatch_chunk[i] = subminibatch[i].size();
      scatter_displs[i] = running_sum;
      running_sum += subminibatch[i].size();
      scatter_minibatch.insert(scatter_minibatch.end(),
                               subminibatch[i].begin(),
                               subminibatch[i].end());
    }
  }

  int32_t my_minibatch_size;
  r = MPI_Scatter(minibatch_chunk.data(), 1, MPI_INT,
                  &my_minibatch_size, 1, MPI_INT,
                  mpi_master_, MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Scatter of minibatch chunks fails");
  nodes_.resize(my_minibatch_size);

  if (mpi_rank_ == mpi_master_) {
    // TODO Master scatters the minibatch nodes over the workers,
    // preferably with consideration for both load balance and locality
    r = MPI_Scatterv(scatter_minibatch.data(),
                     minibatch_chunk.data(),
                     scatter_displs.data(),
                     MPI_INT,
                     nodes_.data(),
                     my_minibatch_size,
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");

  } else {
    r = MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     nodes_.data(), my_minibatch_size, MPI_INT,
                     mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");
  }

  if (true) {
    std::cerr << "Master gives me minibatch nodes[" << my_minibatch_size <<
      "] ";
    for (auto n : nodes_) {
      std::cerr << n << " ";
    }
    std::cerr << std::endl;
  }

  if (! REPLICATED_NETWORK) {
    ScatterSubGraph(subminibatch);
  }

  return edgeSample;
}


void MCMCSamplerStochasticDistributed::update_phi(
    std::vector<std::vector<double>>* phi_node) {
  std::vector<double*> pi_node;
  std::vector<double*> pi_neighbor;
  std::vector<int32_t> flat_neighbors;

  double eps_t  = a * std::pow(1 + step_count / b, -c);	// step size
  // double eps_t = std::pow(1024+step_count, -0.5);

  for (::size_t chunk_start = 0;
       chunk_start < nodes_.size();
       chunk_start += max_minibatch_chunk_) {
    ::size_t chunk = std::min(max_minibatch_chunk_,
                              nodes_.size() - chunk_start);

    std::vector<int32_t> chunk_nodes(nodes_.begin() + chunk_start,
                                     nodes_.begin() + chunk_start + chunk);

    // ************ load minibatch node pi from D-KV store **************
    t_load_pi_minibatch_.start();
    pi_node.resize(chunk_nodes.size());
    d_kv_store_->ReadKVRecords(pi_node, chunk_nodes, DKV::RW_MODE::READ_ONLY);
    t_load_pi_minibatch_.stop();

    // ************ sample neighbor nodes in parallel at each host ******
    // std::cerr << "Sample neighbor nodes" << std::endl;
    t_sample_neighbor_nodes_.start();
    pi_neighbor.resize(chunk_nodes.size() * real_num_node_sample());
    flat_neighbors.resize(chunk_nodes.size() * real_num_node_sample());
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < chunk_nodes.size(); ++i) {
      Vertex node = chunk_nodes[i];
      // sample a mini-batch of neighbors
      auto rng = neighbor_sample_rng_[omp_get_thread_num()];
      NeighborSet neighbors = sample_neighbor_nodes(num_node_sample, node,
                                                    rng);
      assert(neighbors.size() == real_num_node_sample());
      // Cannot use flat_neighbors.insert() because it may (concurrently)
      // attempt to resize flat_neighbors.
      ::size_t j = i * real_num_node_sample();
      for (auto n : neighbors) {
        memcpy(flat_neighbors.data() + j, &n, sizeof n);
        ++j;
      }
    }
    t_sample_neighbor_nodes_.stop();

    // ************ load neighor pi from D-KV store **********
    t_load_pi_neighbor_.start();
    d_kv_store_->ReadKVRecords(pi_neighbor,
                               flat_neighbors,
                               DKV::RW_MODE::READ_ONLY);
    t_load_pi_neighbor_.stop();

    t_update_phi_.start();
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < chunk_nodes.size(); ++i) {
      Vertex node = chunk_nodes[i];
      update_phi_node(chunk_start + i, node, pi_node[i],
                      flat_neighbors.begin() + i * real_num_node_sample(),
                      pi_neighbor.begin() + i * real_num_node_sample(),
                      eps_t, phi_update_rng_[omp_get_thread_num()],
                      &(*phi_node)[chunk_start + i]);
    }
    t_update_phi_.stop();
  }
}


void MCMCSamplerStochasticDistributed::update_phi_node(
    ::size_t index, Vertex i, const double* pi_node,
    const std::vector<int32_t>::iterator &neighbors,
    const std::vector<double*>::iterator &pi,
    double eps_t, Random::Random* rnd,
    std::vector<double>* phi_node	// out parameter
    ) {
  if (omp_get_thread_num() == 0) {
    std::cerr << __func__ << "(): omp num threads " <<
      omp_get_num_threads() << std::endl;
  }
  if (false) {
    std::cerr << "update_phi pre ";
    std::cerr << "phi[" << i << "] ";
    for (::size_t k = 0; k < K; ++k) {
      std::cerr << std::fixed << std::setprecision(12) << (pi_node[k] * pi_node[K]) << " ";
    }
    std::cerr << std::endl;
    std::cerr << "pi[" << i << "] ";
    for (::size_t k = 0; k < K; ++k) {
      std::cerr << std::fixed << std::setprecision(12) << pi_node[k] << " ";
    }
    std::cerr << std::endl;
    for (::size_t ix = 0; ix < real_num_node_sample(); ++ix) {
      int32_t neighbor = neighbors[ix];
      std::cerr << "pi[" << neighbor << "] ";
      for (::size_t k = 0; k < K; ++k) {
        std::cerr << std::fixed << std::setprecision(12) << pi[ix][k] << " ";
      }
      std::cerr << std::endl;
      ++ix;
    }
  }

  double phi_i_sum = pi_node[K];
  std::vector<double> grads(K, 0.0);	// gradient for K classes

  for (::size_t ix = 0; ix < real_num_node_sample(); ++ix) {
    int32_t neighbor = neighbors[ix];
    if (i != neighbor) {
      int y_ab = 0;		// observation
      if (omp_get_max_threads() == 1) {
        t_update_phi_in_.start();
      }
      if (REPLICATED_NETWORK) {
        Edge edge(std::min(i, neighbor), std::max(i, neighbor));
        if (edge.in(network.get_linked_edges())) {
          y_ab = 1;
        }
      } else {
        Edge edge(index, neighbor);
        if (local_network_.find(edge)) {
          y_ab = 1;
        }
      }
      if (omp_get_max_threads() == 1) {
        t_update_phi_in_.stop();
      }

      std::vector<double> probs(K);
      double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
      for (::size_t k = 0; k < K; ++k) {
        double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
        probs[k] = pi_node[k] * (pi[ix][k] * f + e);
      }

      double prob_sum = np::sum(probs);
      // std::cerr << std::fixed << std::setprecision(12) << "node " << i <<
      //    " neighb " << neighbor << " prob_sum " << prob_sum <<
      //    " phi_i_sum " << phi_i_sum <<
      //    " #sample " << real_num_node_sample() << std::endl;
      for (::size_t k = 0; k < K; ++k) {
        grads[k] += ((probs[k] / prob_sum) / pi_node[k] - 1.0) / phi_i_sum;
      }
    } else {
      std::cerr << "Skip self loop <" << i << "," << neighbor << ">" << std::endl;
    }
  }

  std::vector<double> noise = rnd->randn(K);	// random gaussian noise.
  double Nn = (1.0 * N) / num_node_sample;
  // update phi for node i
  for (::size_t k = 0; k < K; ++k) {
    double phi_node_k = pi_node[k] * phi_i_sum;
    (*phi_node)[k] = std::abs(phi_node_k + eps_t / 2 * (alpha - phi_node_k +
                                                        Nn * grads[k]) +
                              sqrt(eps_t * phi_node_k) * noise[k]);
  }
}


void MCMCSamplerStochasticDistributed::update_beta(
    const MinibatchSet &mini_batch, double scale) {

  t_beta_zero_.start();
#pragma omp parallel for
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    for (::size_t k = 0; k < K; ++k) {
      grads_beta_[i][k][0] = 0.0;
      grads_beta_[i][k][1] = 0.0;
    }
  }
  // sums = np.sum(self.__theta,1)
  std::vector<double> theta_sum(theta.size());
  std::transform(theta.begin(), theta.end(), theta_sum.begin(),
                 np::sum<double>);
  t_beta_zero_.stop();

  t_beta_rank_.start();
  // FIXME: already did the nodes_in_batch() -- only the ranking remains
  std::unordered_map<Vertex, Vertex> node_rank;
  std::vector<Vertex> nodes;
  for (auto e : mini_batch) {
    Vertex i = e.first;
    Vertex j = e.second;
    if (node_rank.find(i) == node_rank.end()) {
      ::size_t next = node_rank.size();
      node_rank[i] = next;
      nodes.push_back(i);
    }
    if (node_rank.find(j) == node_rank.end()) {
      ::size_t next = node_rank.size();
      node_rank[j] = next;
      nodes.push_back(j);
    }
    assert(node_rank.size() == nodes.size());
  }
  t_beta_rank_.stop();

  t_load_pi_beta_.start();
  std::vector<double*> pi(node_rank.size());
  d_kv_store_->ReadKVRecords(pi, nodes, DKV::RW_MODE::READ_ONLY);
  t_load_pi_beta_.stop();

  // update gamma, only update node in the grad
  double eps_t = a * std::pow(1.0 + step_count / b, -c);
  //double eps_t = std::pow(1024+step_count, -0.5);
  t_beta_calc_grads_.start();
  std::vector<Edge> v_mini_batch(mini_batch.begin(), mini_batch.end());
#pragma omp parallel for // num_threads (12)
  for (::size_t e = 0; e < v_mini_batch.size(); ++e) {
    const auto *edge = &v_mini_batch[e];
    std::vector<double> probs(K);

    int y = 0;
    if (edge->in(network.get_linked_edges())) {
      y = 1;
    }
    Vertex i = node_rank[edge->first];
    Vertex j = node_rank[edge->second];

    double pi_sum = 0.0;
    for (::size_t k = 0; k < K; ++k) {
      // Note: this is the KV-store cached pi, not the Learner item
      pi_sum += pi[i][k] * pi[j][k];
      double f = pi[i][k] * pi[j][k];
      if (y == 1) {
        probs[k] = beta[k] * f;
      } else {
        probs[k] = (1.0 - beta[k]) * f;
      }
    }

    double prob_0 = ((y == 1) ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
    double prob_sum = np::sum(probs) + prob_0;
    for (::size_t k = 0; k < K; ++k) {
      double f = probs[k] / prob_sum;
      double one_over_theta_sum = 1.0 / theta_sum[k];
      grads_beta_[omp_get_thread_num()][k][0] += f * ((1 - y) / theta[k][0] -
                                                      one_over_theta_sum);
      grads_beta_[omp_get_thread_num()][k][1] += f * (y / theta[k][1] -
                                                      one_over_theta_sum);
    }
  }
  t_beta_calc_grads_.stop();

  t_beta_sum_grads_.start();
#pragma omp parallel for
  for (::size_t k = 0; k < K; ++k) {
    for (int i = 1; i < omp_get_max_threads(); ++i) {
      grads_beta_[0][k][0] += grads_beta_[i][k][0];
      grads_beta_[0][k][1] += grads_beta_[i][k][1];
    }
  }
  t_beta_sum_grads_.stop();

  t_beta_update_theta_.start();
  // update theta

  // random noise.
  std::vector<std::vector<double> > noise =
    rng_.random(SourceAwareRandom::BETA_UPDATE)->randn(K, 2);
#pragma omp parallel for
  for (::size_t k = 0; k < K; ++k) {
    for (::size_t i = 0; i < 2; ++i) {
      double f = std::sqrt(eps_t * theta[k][i]);
      theta[k][i] = std::abs(theta[k][i] +
                             eps_t / 2.0 * (eta[i] - theta[k][i] +
                                            scale * grads_beta_[0][k][i]) +
                             f * noise[k][i]);
    }
  }

  std::vector<std::vector<double> > temp(theta.size(),
                                         std::vector<double>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<double>(1));

  d_kv_store_->PurgeKVRecords();
  t_beta_update_theta_.stop();

}


void MCMCSamplerStochasticDistributed::reduce_plus(const perp_accu &in,
                                                   perp_accu* accu) {
  int r;
  uint64_t count[2] = { in.link.count, in.non_link.count };
  double likelihood[2] = { in.link.likelihood, in.non_link.likelihood };

  r = MPI_Allreduce(MPI_IN_PLACE, count, 2, MPI_UNSIGNED_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
  mpi_error_test(r, "Reduce/plus of perplexity counts fails");
  r = MPI_Allreduce(MPI_IN_PLACE, likelihood, 2, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
  mpi_error_test(r, "Reduce/plus of perplexity likelihoods fails");

  accu->link.count = count[0];
  accu->non_link.count = count[1];
  accu->link.likelihood = likelihood[0];
  accu->non_link.likelihood = likelihood[1];
}


/**
 * calculate the perplexity for data.
 * perplexity defines as exponential of negative average log likelihood. 
 * formally:
 *     ppx = exp(-1/N * \sum){i}^{N}log p(y))
 * 
 * we calculate average log likelihood for link and non-link separately, with the 
 * purpose of weighting each part proportionally. (the reason is that we sample 
 * the equal number of link edges and non-link edges for held out data and test data,
 * which is not true representation of actual data set, which is extremely sparse.
 */
double MCMCSamplerStochasticDistributed::cal_perplexity_held_out() {

  for (auto & a : perp_.accu_) {
    a.link.reset();
    a.non_link.reset();
  }

  for (::size_t chunk_start = 0;
       chunk_start < perp_.data_.size();
       chunk_start += max_perplexity_chunk_) {
    ::size_t chunk = std::min(max_perplexity_chunk_,
                              perp_.data_.size() - chunk_start);

    // chunk_size is about edges; nodes are at 2i and 2i+1
    std::vector<int32_t> chunk_nodes(perp_.nodes_.begin() + 2 * chunk_start,
                                     perp_.nodes_.begin() + 2 * (chunk_start + chunk));

    t_load_pi_perp_.start();
    d_kv_store_->ReadKVRecords(perp_.pi_, chunk_nodes, DKV::RW_MODE::READ_ONLY);
    t_load_pi_perp_.stop();

    t_cal_edge_likelihood_.start();
#pragma omp parallel for
    for (::size_t i = chunk_start; i < chunk_start + chunk; ++i) {
      const auto& edge_in = perp_.data_[i];
      // the index into the nodes/pi vectors is double the index into the
      // edge vector (+ 1)
      Vertex a = 2 * (i - chunk_start);
      Vertex b = 2 * (i - chunk_start) + 1;
      double edge_likelihood = cal_edge_likelihood(perp_.pi_[a], perp_.pi_[b],
                                                   edge_in.is_edge, beta);
      if (std::isnan(edge_likelihood)) {
        std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
      }

      //cout<<"AVERAGE COUNT: " <<average_count;
      ppx_for_heldout[i] = (ppx_for_heldout[i] * (average_count-1) + edge_likelihood)/(average_count);
      // std::cout << std::fixed << std::setprecision(12) << e <<
      //   " in? " << (e.in(network.get_linked_edges()) ? "True" : "False") <<
      //   " -> " << edge_likelihood << " av. " << average_count <<
      //   " ppx[" << i << "] " << ppx_for_heldout[i] << std::endl;
      if (edge_in.is_edge) {
        perp_.accu_[omp_get_thread_num()].link.count++;
        perp_.accu_[omp_get_thread_num()].link.likelihood += std::log(ppx_for_heldout[i]);
        //link_likelihood += edge_likelihood;

        if (std::isnan(perp_.accu_[omp_get_thread_num()].link.likelihood)){
          std::cerr << "link_likelihood is NaN; potential bug" << std::endl;
        }
      } else {
        perp_.accu_[omp_get_thread_num()].non_link.count++;
        //perp_.accu_[omp_get_thread_num()].non_link.likelihood +=
        //  edge_likelihood;
        perp_.accu_[omp_get_thread_num()].non_link.likelihood +=
          std::log(ppx_for_heldout[i]);
        if (std::isnan(perp_.accu_[omp_get_thread_num()].non_link.likelihood)){
          std::cerr << "non_link_likelihood is NaN; potential bug" << std::endl;
        }
      }
    }
  }

  for (auto i = 1; i < omp_get_max_threads(); ++i) {
    perp_.accu_[0].link.count += perp_.accu_[i].link.count;
    perp_.accu_[0].link.likelihood += perp_.accu_[i].link.likelihood;
    perp_.accu_[0].non_link.count += perp_.accu_[i].non_link.count;
    perp_.accu_[0].non_link.likelihood += perp_.accu_[i].non_link.likelihood;
  }

  t_cal_edge_likelihood_.stop();

  t_purge_pi_perp_.start();
  d_kv_store_->PurgeKVRecords();
  t_purge_pi_perp_.stop();

  // std::cout << std::setprecision(12) << "ratio " << link_ratio <<
  //   " count: link " << link_count << " " << link_likelihood <<
  //   " non-link " << non_link_count << " " << non_link_likelihood <<
  //   std::endl;

  // weight each part proportionally.
  /*
     avg_likelihood = self._link_ratio*(link_likelihood/link_count) + \
     (1-self._link_ratio)*(non_link_likelihood/non_link_count)
     */

  perp_accu accu;
  t_reduce_perp_.start();
  reduce_plus(perp_.accu_[0], &accu);
  t_reduce_perp_.stop();

  // direct calculation.
  double avg_likelihood = 0.0;
  if (accu.link.count + accu.non_link.count != 0){
    avg_likelihood = (accu.link.likelihood + accu.non_link.likelihood) /
      (accu.link.count + accu.non_link.count);
  }
  if (true && mpi_rank_ == mpi_master_) {
    double avg_likelihood1 = link_ratio *
      (accu.link.likelihood / accu.link.count) +
      (1.0 - link_ratio) * (accu.non_link.likelihood / accu.non_link.count);
    std::cout << std::fixed << std::setprecision(12) <<
      avg_likelihood << " " << (accu.link.likelihood / accu.link.count) <<
      " " << accu.link.count << " " <<
      (accu.non_link.likelihood / accu.non_link.count) << " " <<
      accu.non_link.count << " " << avg_likelihood1 << std::endl;
    // std::cout << "perplexity score is: " << exp(-avg_likelihood) <<
    //   std::endl;
  }

  // return std::exp(-avg_likelihood);

  //if (step_count > 1000000)
  average_count = average_count + 1;
  if (mpi_rank_ == mpi_master_) {
    std::cout << "average_count is: " << average_count << " ";
  }

  return (-avg_likelihood);
}


int MCMCSamplerStochasticDistributed::node_owner(Vertex node) const {
  if (master_is_worker_) {
    return node % mpi_size_;
  } else {
    return 1 + (node % (mpi_size_ - 1));
  }
}


void MCMCSamplerStochasticDistributed::mpi_error_test(
    int r, const std::string &message) {
  if (r != MPI_SUCCESS) {
    throw MCMCException("MPI error " + r + message);
  }
}

}	// namespace learning
}	// namespace mcmc