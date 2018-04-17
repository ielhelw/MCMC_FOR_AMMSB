#include <mcmc/learning/mcmc_sampler_stochastic_distr.h>

#include <cinttypes>
#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "mcmc/exception.h"
#include "mcmc/config.h"

#include "mcmc/fixed-size-set.h"

#ifdef MCMC_SINGLE_PRECISION
#  define FLOATTYPE_MPI MPI_FLOAT
#else
#  define FLOATTYPE_MPI MPI_DOUBLE
#endif

#ifdef MCMC_ENABLE_DISTRIBUTED
#  include <mpi.h>
#else
#  include "mock_mpi.h"
#endif

#include "dkvstore/DKVStoreFile.h"
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

void LocalNetwork::swap(::size_t from, ::size_t to) {
  linked_edges_[from].swap(linked_edges_[to]);
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
// class ChunkPipeline
//
// **************************************************************************

void ChunkPipeline::EnqueueChunk(PiChunk* chunk) {
  boost::unique_lock<boost::mutex> lock(lock_);
  if (buffer_[client_enq_].state_ != PIPELINE_STATE::FREE) {
    throw MCMCException("Buffer should be free but it isn't");
  }

  // std::cerr << "Client: enqueue buffer[" << client_enq_ << "] for filling" << std::endl;
  buffer_[client_enq_].chunk_ = chunk;
  buffer_[client_enq_].state_ = PIPELINE_STATE::FILL_REQUESTED;
  buffer_[client_enq_].fill_requested_.notify_one();
  client_enq_ = (client_enq_ + 1) % num_buffers();
}


PiChunk* ChunkPipeline::AwaitChunkFilled() {
  boost::unique_lock<boost::mutex> lock(lock_);
  // std::cerr << "Client: await buffer[" << client_deq_ << "] as full" << std::endl;
  while (buffer_[client_deq_].state_ != PIPELINE_STATE::FILLED) {
    buffer_[client_deq_].fill_completed_.wait(lock);
  }
  // std::cerr << "Client: got buffer[" << client_deq_ << "], now compute" << std::endl;
  PiChunk* pi_chunk = buffer_[client_deq_].chunk_;
  client_clear_ = client_deq_;
  client_deq_ = (client_deq_ + 1) % num_buffers();
  return pi_chunk;
}


PiChunk* ChunkPipeline::DequeueChunk() {
  boost::unique_lock<boost::mutex> lock(lock_);
  while (buffer_[server_deq_].state_ != PIPELINE_STATE::FILL_REQUESTED) {
    if (buffer_[server_deq_].state_ == PIPELINE_STATE::STOP) {
      return NULL;
    }
    buffer_[server_deq_].fill_requested_.wait(lock);
  }
  // std::cerr << "Server: dequeue buffer[" << server_deq_ << "] for filling" << std::endl;

  PiChunk* chunk = buffer_[server_deq_].chunk_;
  chunk->buffer_index_ = server_deq_ % num_buffers();
  buffer_[server_deq_].state_ = PIPELINE_STATE::FILLING;
  server_deq_ = (server_deq_ + 1) % num_buffers();

  return chunk;
}


void ChunkPipeline::NotifyChunkFilled() {
  if (buffer_[server_complete_].state_ != PIPELINE_STATE::FILLING) {
    throw MCMCException("Buffer state should be FILLING");
  }
  boost::unique_lock<boost::mutex> lock(lock_);
  // std::cerr << "Server: notify buffer[" << server_complete_ << "] as full" << std::endl;

  buffer_[server_complete_].state_ = PIPELINE_STATE::FILLED;
  buffer_[server_complete_].fill_completed_.notify_one();
  server_complete_ = (server_complete_ + 1) % num_buffers();
}


void ChunkPipeline::ReleaseChunk() {
  if (buffer_[client_clear_].state_ != PIPELINE_STATE::FILLED) {
    throw MCMCException("Buffer state should be FILLED");
  }
  boost::unique_lock<boost::mutex> lock(lock_);
  // std::cerr << "Client: purge buffer[" << client_clear_ << "] as full" << std::endl;
  buffer_[client_clear_].state_ = PIPELINE_STATE::FREE;
  d_kv_store_.PurgeKVRecords(client_clear_);
  client_clear_ = (client_clear_ + 1) % num_buffers();
}


void ChunkPipeline::Stop() {
  boost::unique_lock<boost::mutex> lock(lock_);
  for (auto& b : buffer_) {
    b.state_ = PIPELINE_STATE::STOP;
    b.fill_requested_.notify_one();
  }
}


::size_t ChunkPipeline::num_buffers() const {
  return buffer_.size();
}


void ChunkPipeline::operator() () {
  std::cerr << "DKV Pipeline server runs" << std::endl;
  PiChunk* chunk;
  t_load_pi_minibatch_     = Timer("      load minibatch pi");
  t_load_pi_neighbor_      = Timer("      load neighbor pi");

  while (true) {
    chunk = DequeueChunk();
    if (chunk == NULL) {
      break;
    }

    // ************ load minibatch node pi from D-KV store **************
    t_load_pi_minibatch_.start();
    d_kv_store_.ReadKVRecords(chunk->buffer_index_, chunk->pi_node_,
                              chunk->chunk_nodes_);
    t_load_pi_minibatch_.stop();

    // ************ load neighbor pi from D-KV store **********
    t_load_pi_neighbor_.start();
    d_kv_store_.ReadKVRecords(chunk->buffer_index_, chunk->pi_neighbor_,
                              chunk->flat_neighbors_);
    t_load_pi_neighbor_.stop();

    NotifyChunkFilled();
  }
}


std::ostream& ChunkPipeline::report(std::ostream& s) const {
  s << t_load_pi_minibatch_ << std::endl;
  s << t_load_pi_neighbor_ << std::endl;
  return s;
}


::size_t ChunkPipeline::GrabFreeBufferIndex() {
  boost::unique_lock<boost::mutex> lock(lock_);
  for (::size_t i = 0; i < buffer_.size(); ++i) {
    if (buffer_[i].state_ == PIPELINE_STATE::FREE) {
      buffer_[i].state_ = PIPELINE_STATE::USE_EXTERN;
      return i;
    }
  }
  throw MCMCException("No free buffers in pi cache");
}


void ChunkPipeline::ReleaseGrabbedBuffer(::size_t index) {
  if (buffer_[index].state_ != PIPELINE_STATE::USE_EXTERN) {
    throw MCMCException("Buffer state should be USE_EXTERN");
  }
  boost::unique_lock<boost::mutex> lock(lock_);
  d_kv_store_.PurgeKVRecords(index);
  buffer_[index].state_ = PIPELINE_STATE::FREE;
}


// **************************************************************************
//
// class PiChunk
//
// **************************************************************************

void PiChunk::swap(PiChunk& x) {
  std::swap(buffer_index_, x.buffer_index_);
  std::swap(start_, x.start_);
  chunk_nodes_.swap(x.chunk_nodes_);
  pi_node_.swap(x.pi_node_);
  flat_neighbors_.swap(x.flat_neighbors_);
  pi_neighbor_.swap(x.pi_neighbor_);
}


// **************************************************************************
//
// class MinibatchSlice
//
// **************************************************************************

void MinibatchSlice::SwapNodeAndGraph(::size_t i, ::size_t with,
                                      PiChunk* c, PiChunk* with_c) {
  // std::cerr << "Node[" << i << "] " << c->chunk_nodes_[i] << " overlaps previous minibatch, swap with [" << with_c->start_ + with << "] " << with_c->chunk_nodes_[with];
  std::swap(c->chunk_nodes_[i], with_c->chunk_nodes_[with]);
  std::swap(nodes_[i], nodes_[with_c->start_ + with]);
  if (! replicated_network_) {
    local_network_.swap(c->start_ + i, with_c->start_ + with);
  }
  // std::cerr << " becomes [" << i << "] " << c->chunk_nodes_[i] << " swap with [" << with_c->start_ + with << "] " << with_c->chunk_nodes_[with] << std::endl;
}


// **************************************************************************
//
// class MinibatchPipeline
//
// **************************************************************************

MinibatchPipeline::MinibatchPipeline(MCMCSamplerStochasticDistributed& sampler,
                                     ::size_t max_minibatch_chunk,
                                     ChunkPipeline& chunk_pipeline,
                                     std::vector<Random::Random*>& rng,
                                     bool replicated_network)
    : sampler_(sampler), chunk_pipeline_(chunk_pipeline),
      max_minibatch_chunk_(max_minibatch_chunk), rng_(rng),
      prev_(2), current_(0), next_(1) {
  minibatch_slice_.resize(3, replicated_network);    // prev, current, next
  t_chunk_minibatch_slice_     = Timer("      create minibatch slice chunks");
  t_reorder_minibatch_overlap_ = Timer("      reorder minibatch overlap");
  t_sample_neighbor_nodes_     = Timer("      sample_neighbor_nodes");
  t_resample_neighbor_nodes_   = Timer("      resample_neighbor_nodes");
}


void MinibatchPipeline::StageNextChunk() {
  MinibatchSlice* mb_slice = &minibatch_slice_[current_];
  if (mb_slice->processed_ >= mb_slice->pi_chunks_.size()) {
    StageNextMinibatch();
  } else {
    chunk_pipeline_.EnqueueChunk(&mb_slice->pi_chunks_[mb_slice->processed_]);
    ++mb_slice->processed_;
  }
}


void MinibatchPipeline::AdvanceMinibatch() {
  prev_ = current_;
  current_ = next_;
  next_ = (next_ + 1) % minibatch_slice_.size();
}


void MinibatchPipeline::StageNextMinibatch() {
  // Receives minibatch slice from master;
  //    samples neighbors for the minibatch slice
  //    builds the chunks so the first chunk has no overlap w/ prev_chunk
  // Receives the full minibatch nodes from master.
  MinibatchSlice* mb_slice = &minibatch_slice_[next_];
  sampler_.deploy_mini_batch(mb_slice);
  CreateMinibatchSliceChunks(mb_slice);
  SampleNeighbors(mb_slice);
  chunk_pipeline_.EnqueueChunk(&mb_slice->pi_chunks_[0]);
  mb_slice->processed_ = 1;
}


bool MinibatchPipeline::PreviousMinibatchOverlap(Vertex one) const {
  const VertexSet& other = minibatch_slice_[current_].full_minibatch_nodes_;
  return (other.find(one) != other.end());
}


/**
 * If the first chunk of the current minibatch has nodes that overlap with the
 * previous minibatch, reorder the minibatch nodes. Note: the entries in the
 * graph must also be reordered to keep consistent.
 */
void MinibatchPipeline::ReorderMinibatchOverlap(
    MinibatchSlice* mb_slice) {
  auto *pi_chunk = &mb_slice->pi_chunks_[0];
  ::size_t last_checked_chunk = mb_slice->pi_chunks_.size() - 1;
  auto *last_pi_chunk = &mb_slice->pi_chunks_[last_checked_chunk];
  ::size_t last_checked_node = last_pi_chunk->chunk_nodes_.size();
  bool first = true;
  for (::size_t i = 0; i < pi_chunk->chunk_nodes_.size(); ++i) {
    if (PreviousMinibatchOverlap(pi_chunk->chunk_nodes_[i])) {
      if (false && first) {
        first = false;
        std::cerr << "Previous minibatch: ";
        for (auto v : minibatch_slice_[current_].full_minibatch_nodes_) {
          std::cerr << v << " ";
        }
        std::cerr << std::endl;
        std::cerr << "Current minibatch: ";
        for (auto v : minibatch_slice_[next_].full_minibatch_nodes_) {
          std::cerr << v << " ";
        }
        std::cerr << std::endl;
        std::cerr << "My chunks:" << std::endl;
        for (auto& c : mb_slice->pi_chunks_) {
          std::cerr << "    ";
          for (auto v : c.chunk_nodes_) {
            std::cerr << v << " ";
          }
          std::cerr << std::endl;
        }
      }
      t_reorder_minibatch_overlap_.start();
      while (true) {
        while (last_checked_node == 0) {
          --last_checked_chunk;
          last_pi_chunk = &mb_slice->pi_chunks_[last_checked_chunk];
          last_checked_node = last_pi_chunk->chunk_nodes_.size();
          if (last_checked_chunk == 0) {
            /* Special case: run out of swappable vertices in higher chunks.
             * Do a different algorithm. */
            if (false) {
              std::cerr << "********************** Cannot swap [" << i << "] " << pi_chunk->chunk_nodes_[i] << std::endl;
              std::cerr << "First chunk in my current minibatch slice: ";
              for (auto v : pi_chunk->chunk_nodes_) {
                std::cerr << v << " ";
              }
              std::cerr << std::endl;
            }

            /*
             * Algorithm:
             *  - swap the current vertex w/ the last nonoverlapping vertex
             *    within this slice, do the same w/ graph entries
             *  - at the end
             *     + insert the overlapping vertices of the first chunk into the
             *       beginning of the second chunk
             *     + decrement the start pointer of the second chunk
             * Border case? the next chunk is empty.
             */
            for (; i < last_checked_node; ++i) {
              if (PreviousMinibatchOverlap(pi_chunk->chunk_nodes_[i])) {
                while (last_checked_node > i) {
                  --last_checked_node;
                  if (! PreviousMinibatchOverlap(pi_chunk->chunk_nodes_[last_checked_node])) {
                    mb_slice->SwapNodeAndGraph(i, last_checked_node, pi_chunk,
                                               pi_chunk);
                    break;
                  }
                }
              }
            }

            last_pi_chunk = &mb_slice->pi_chunks_[1];
            ::size_t n_swap = pi_chunk->chunk_nodes_.size() - last_checked_node;
            if (last_pi_chunk->chunk_nodes_.size() == 0) {
              last_pi_chunk->start_ = 0;
            } else {
              assert(last_pi_chunk->start_ >= n_swap);
              last_pi_chunk->start_ -= n_swap;
            }
            last_pi_chunk->chunk_nodes_.insert(
              last_pi_chunk->chunk_nodes_.begin(),
              pi_chunk->chunk_nodes_.begin() +
                (pi_chunk->chunk_nodes_.size() - n_swap),
              pi_chunk->chunk_nodes_.end());
            pi_chunk->chunk_nodes_.erase(pi_chunk->chunk_nodes_.begin() + last_checked_node, pi_chunk->chunk_nodes_.end());

            last_pi_chunk->pi_node_.resize(last_pi_chunk->chunk_nodes_.size());
            pi_chunk->pi_node_.resize(pi_chunk->chunk_nodes_.size());

            if (false) {
              std::cerr << "After: my chunks:" << std::endl;
              for (auto& c : mb_slice->pi_chunks_) {
                std::cerr << "    start=" << c.start_ << " ";
                for (auto v : c.chunk_nodes_) {
                  std::cerr << v << " ";
                }
                std::cerr << std::endl;
              }
            }
            t_reorder_minibatch_overlap_.stop();
            return;
          }
        }

        --last_checked_node;
        if (! PreviousMinibatchOverlap(last_pi_chunk->chunk_nodes_[last_checked_node])) {
          mb_slice->SwapNodeAndGraph(i, last_checked_node, pi_chunk,
                                     last_pi_chunk);
          break;
        }
      }
      t_reorder_minibatch_overlap_.stop();
    }
  }
}


void MinibatchPipeline::CreateMinibatchSliceChunks(
    MinibatchSlice* mb_slice) {
  t_chunk_minibatch_slice_.start();

  ::size_t num_chunks = (mb_slice->nodes_.size() + max_minibatch_chunk_ - 1) /
                          max_minibatch_chunk_;
  if (num_chunks <= 1) {
    num_chunks = 2;
  }
  ::size_t chunk_size = (mb_slice->nodes_.size() + num_chunks - 1) / num_chunks;
  mb_slice->pi_chunks_.resize(num_chunks);

  ::size_t chunk_start = 0;
  for (::size_t b = 0; b < num_chunks; b++) {
    ::size_t chunk = std::min(chunk_size,
                              mb_slice->nodes_.size() - chunk_start);

    if (false && chunk <= 1) {
      if (b == 0) {
        std::cerr << "Minibatch slice size " << mb_slice->nodes_.size() << " chunk size";
      }
      std::cerr << " " << chunk;
      if (b == num_chunks - 1) {
        std::cerr << std::endl;
      }
    }
    auto *pi_chunk = &mb_slice->pi_chunks_[b];
    const auto &chunk_begin = mb_slice->nodes_.begin() + chunk_start;
    pi_chunk->chunk_nodes_.assign(chunk_begin, chunk_begin + chunk);
    pi_chunk->pi_node_.resize(pi_chunk->chunk_nodes_.size());
    pi_chunk->start_ = chunk_start;
    chunk_start += chunk;
  }
  assert(chunk_start == mb_slice->nodes_.size());

  ReorderMinibatchOverlap(mb_slice);
  t_chunk_minibatch_slice_.stop();
}


// ************ sample neighbor nodes in parallel at each host ******
void MinibatchPipeline::SampleNeighbors(
    MinibatchSlice* mb_slice) {
  t_sample_neighbor_nodes_.start();
  ::size_t num_chunks = mb_slice->pi_chunks_.size();

  for (::size_t b = 0; b < num_chunks; b++) {
    auto *pi_chunk = &mb_slice->pi_chunks_[b];
    pi_chunk->pi_neighbor_.resize(pi_chunk->chunk_nodes_.size() *
                                  sampler_.real_num_node_sample());
    pi_chunk->flat_neighbors_.resize(pi_chunk->chunk_nodes_.size() *
                                     sampler_.real_num_node_sample());

    // std::cerr << "Sample neighbor nodes" << std::endl;
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < pi_chunk->chunk_nodes_.size(); ++i) {
      if (! SampleNeighborSet(pi_chunk, i)) {
        t_resample_neighbor_nodes_.start();
        while (! SampleNeighborSet(pi_chunk, i)) {
          // overlap, sample again
          // std::cerr << "Overlap in neighbor sample " << i << ", do it again" << std::endl;
        }
        t_resample_neighbor_nodes_.stop();
      }
    }
  }
  t_sample_neighbor_nodes_.stop();
}


bool MinibatchPipeline::SampleNeighborSet(PiChunk* pi_chunk, ::size_t i) {
  Vertex node = pi_chunk->chunk_nodes_[i];
  // sample a mini-batch of neighbors
  auto rng = rng_[omp_get_thread_num()];
  ::size_t p = sampler_.real_num_node_sample();
  FixedSizeSet neighbors(p);
  while (neighbors.size() < p) {
    const Vertex neighborId = rng->randint(0, sampler_.N - 1); 
    if (neighborId != node &&
          neighbors.find(neighborId) == neighbors.end()) {
      const Edge edge = Edge(std::min(node, neighborId),
                             std::max(node, neighborId));
      if (! edge.in(sampler_.held_out_test())) {
        neighbors.insert(neighborId);
      }
    }
  }

  // Cannot use flat_neighbors_.insert() because it may (concurrently)
  // attempt to resize flat_neighbors_.
  ::size_t j = i * p;
  for (auto n : neighbors) {
    if (i == 0 && PreviousMinibatchOverlap(n)) {
      // std::cerr << "neighbor " << n << " is also in previous minibatch" << std::endl;
      return false;
    }
    pi_chunk->flat_neighbors_[j] = n;
    ++j;
  }
  return true;
}


std::ostream& MinibatchPipeline::report(std::ostream& s) const {
  s << t_chunk_minibatch_slice_ << std::endl;
  s << t_reorder_minibatch_overlap_ << std::endl;
  s << t_sample_neighbor_nodes_ << std::endl;
  s << t_resample_neighbor_nodes_ << std::endl;
  return s;
}


// **************************************************************************
//
// class MCMCSamplerStochasticDistributed
//
// **************************************************************************
MCMCSamplerStochasticDistributed::MCMCSamplerStochasticDistributed(
    const Options &args) : MCMCSamplerStochastic(args), mpi_master_(0) {
  stats_print_interval_ = 64 * 1024;

  t_load_network_          = Timer("  load network graph");
  t_init_dkv_              = Timer("  initialize DKV store");
  t_populate_pi_           = Timer("  populate pi");
  t_outer_                 = Timer("  iteration");
  t_deploy_minibatch_      = Timer("    deploy minibatch");
  t_mini_batch_            = Timer("      sample_mini_batch");
  t_scatter_subgraph_      = Timer("      scatter subgraph");
  t_scatter_subgraph_marshall_edge_count_ = Timer("        marshall edge count");
  t_scatter_subgraph_scatterv_edge_count_ = Timer("        scatterv edges");
  t_scatter_subgraph_marshall_edges_      = Timer("        marshall edges");
  t_scatter_subgraph_scatterv_edges_      = Timer("        scatterv edges");
  t_scatter_subgraph_unmarshall_          = Timer("        unmarshall edges");
  t_nodes_in_mini_batch_   = Timer("      nodes_in_mini_batch");
  t_broadcast_theta_beta_  = Timer("    broadcast theta/beta");
  t_update_phi_pi_         = Timer("    update_phi_pi");
  t_update_phi_            = Timer("      update_phi");
  t_barrier_phi_           = Timer("      barrier after update phi");
  t_update_pi_             = Timer("      update_pi");
  t_store_pi_minibatch_    = Timer("      store minibatch pi");
  t_barrier_pi_            = Timer("      barrier after update pi");
  t_update_beta_           = Timer("    update_beta_theta");
  t_beta_zero_             = Timer("      zero beta grads");
  t_beta_rank_             = Timer("      rank minibatch nodes");
  t_beta_calc_grads_       = Timer("      beta calc grads");
  t_beta_sum_grads_        = Timer("      beta sum grads");
  t_beta_reduce_grads_     = Timer("      beta reduce(+) grads");
  t_beta_update_theta_     = Timer("      update theta");
  t_load_pi_beta_          = Timer("      load pi update_beta");
  t_perplexity_            = Timer("  perplexity");
  t_load_pi_perp_          = Timer("      load perplexity pi");
  t_cal_edge_likelihood_   = Timer("      calc edge likelihood");
  t_purge_pi_perp_         = Timer("      purge perplexity pi");
  t_reduce_perp_           = Timer("      reduce/plus perplexity");
  Timer::setTabular(true);
}


MCMCSamplerStochasticDistributed::~MCMCSamplerStochasticDistributed() {
  for (auto &p : pi_update_) {
    delete[] p;
  }

  (void)MPI_Finalize();
}

void MCMCSamplerStochasticDistributed::InitSlaveState(const NetworkInfo &info,
                                                      ::size_t world_rank) {
  Learner::InitRandom(world_rank);
  network = Network(info);
  Learner::Init(false);
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
    InitSlaveState(info, mpi_rank_);
  }
}


void MCMCSamplerStochasticDistributed::BroadcastHeldOut() {
  int r;
  int32_t my_held_out_size;

  if (mpi_rank_ == mpi_master_) {
    std::vector<int32_t> count(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> displ(mpi_size_);	// FIXME: lift to class

    if (args_.REPLICATED_NETWORK) {
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
      for (::size_t i = surplus; i < static_cast< ::size_t>(mpi_size_); ++i) {
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
    mpi_error_test(r, "MPI_Scatterv of held-out set data fails");
    std::cerr << "My held-out size " << my_held_out_size << std::endl;
  }

  /* Implies broadcast of the keys in held_out set */
  GoogleHashEdgeSet held_out(network.get_held_out_set(),
                             mpi_rank_, mpi_master_, MPI_COMM_WORLD);
  /* Implies broadcast of the keys in test set */
  GoogleHashEdgeSet test(network.get_test_set(),
                         mpi_rank_, mpi_master_, MPI_COMM_WORLD);

  held_out_test_.insert(held_out.begin(), held_out.end());
  held_out_test_.insert(test.begin(), test.end());

  std::cerr << "Held-out+test size " << held_out_test_.size() << std::endl;
  std::cerr << "Test size " << network.get_test_set().size() << std::endl;
  std::cerr << "Held-out size " << network.get_held_out_set().size() << std::endl;
}


void MCMCSamplerStochasticDistributed::MasterAwareLoadNetwork() {
  if (args_.REPLICATED_NETWORK) {
    LoadNetwork(mpi_rank_, false);
  } else {
    if (mpi_rank_ == mpi_master_) {
      LoadNetwork(mpi_rank_, false);
    }
    BroadcastNetworkInfo();
    // No need to broadcast the Network aux stuff, fan_out_cumul_distro and
    // cumulative_edges: it is used at the master only
  }
  BroadcastHeldOut();
}


void MCMCSamplerStochasticDistributed::InitDKVStore() {
  t_init_dkv_.start();

  std::cerr << "Use D-KV store type " << args_.dkv_type << std::endl;
  switch (args_.dkv_type) {
  case DKV::TYPE::FILE:
    d_kv_store_ = std::unique_ptr<DKV::DKVFile::DKVStoreFile>(
                    new DKV::DKVFile::DKVStoreFile(args_.getRemains()));
    break;
#ifdef MCMC_ENABLE_RAMCLOUD
  case DKV::TYPE::RAMCLOUD:
    d_kv_store_ = std::unique_ptr<DKV::DKVRamCloud::DKVStoreRamCloud>(
                    new DKV::DKVRamCloud::DKVStoreRamCloud(args_.getRemains()));
    break;
#endif
#ifdef MCMC_ENABLE_RDMA
  case DKV::TYPE::RDMA:
    d_kv_store_ = std::unique_ptr<DKV::DKVRDMA::DKVStoreRDMA>(
                    new DKV::DKVRDMA::DKVStoreRDMA(args_.getRemains()));
    break;
#endif
  }

  num_buffers_ = args_.num_buffers_;
  if (args_.max_pi_cache_entries_ == 0) {
    std::ifstream meminfo("/proc/meminfo");
    int64_t mem_total = -1;
    while (meminfo.good()) {
      char buffer[256];
      char* colon;

      meminfo.getline(buffer, sizeof buffer);
      if (strncmp("MemTotal", buffer, 8) == 0 &&
           (colon = strchr(buffer, ':')) != 0) {
        if (sscanf(colon + 2, "%ld", &mem_total) != 1) {
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
    ::size_t pi_total = (1024 * mem_total) / ((K + 1) * sizeof(Float));
    // args_.max_pi_cache_entries_ = num_buffers_ * pi_total / 32;
    args_.max_pi_cache_entries_ = pi_total / 32;
    std::cerr << "mem_total " << mem_total << " pi_total " << pi_total << " max pi cache entries " << args_.max_pi_cache_entries_ << std::endl;
  }

  // Calculate DKV store buffer requirements
  max_minibatch_nodes_ = network.max_minibatch_nodes_for_strategy(
                            mini_batch_size, strategy);
  ::size_t workers;
  if (master_is_worker_) {
    workers = mpi_size_;
  } else {
    workers = mpi_size_ - 1;
  }

  // pi cache hosts chunked subset of minibatch nodes + their neighbors
  max_minibatch_chunk_ = args_.max_pi_cache_entries_ / num_buffers_ / (1 + real_num_node_sample());
  max_dkv_write_entries_ = (max_minibatch_nodes_ + workers - 1) / workers;
  ::size_t max_my_minibatch_nodes = std::min(max_minibatch_chunk_,
                                             max_dkv_write_entries_);
  ::size_t max_minibatch_neighbors = max_my_minibatch_nodes *
                                      real_num_node_sample();

  // for perplexity, cache pi for both vertexes of each edge
  max_perplexity_chunk_ = args_.max_pi_cache_entries_ / (2 * num_buffers_);
  ::size_t num_perp_nodes = 2 * (network.get_held_out_size() +
                                 mpi_size_ - 1) / mpi_size_;
  ::size_t max_my_perp_nodes = std::min(2 * max_perplexity_chunk_,
                                        num_perp_nodes);

  // must cache pi[minibatch slice] for update_beta
  ::size_t max_beta_nodes = (max_minibatch_nodes_ + mpi_size_ - 1) / mpi_size_;
  max_minibatch_neighbors = std::max(max_minibatch_neighbors, max_beta_nodes);
  if (max_minibatch_neighbors > args_.max_pi_cache_entries_) {
    throw MCMCException("pi cache cannot contain pi[minibatch] for beta, "
                        "refactor so update_beta is chunked");
  }

  max_dkv_pi_cache_ = std::max(max_my_minibatch_nodes +
                               max_minibatch_neighbors,
                               max_my_perp_nodes);

  std::cerr << "minibatch size param " << mini_batch_size <<
    " max " << max_minibatch_nodes_ <<
    " my max " << max_my_minibatch_nodes <<
    " chunk " << max_minibatch_chunk_ <<
    " #neighbors(total) " << max_minibatch_neighbors <<
    " cache max entries " << max_dkv_pi_cache_ <<
    " computed max pi cache entries " << args_.max_pi_cache_entries_ <<
    std::endl;
  std::cerr << "perplexity nodes total " <<
    (network.get_held_out_size() * 2) <<
    " local " << num_perp_nodes <<
    " mine " << max_my_perp_nodes <<
    " chunk " << max_perplexity_chunk_ << std::endl;
  std::cerr << "phi pipeline depth " << num_buffers_ << std::endl;

  d_kv_store_->Init(K + 1, N, num_buffers_, max_dkv_pi_cache_,
                    max_dkv_write_entries_);
  t_init_dkv_.stop();

  master_hosts_pi_ = d_kv_store_->include_master();

  std::cerr << "Master is " << (master_is_worker_ ? "" : "not ") <<
    "a worker, does " << (master_hosts_pi_ ? "" : "not ") <<
    "host pi values" << std::endl;
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

  std::cerr << "MPI_Init() done, rank " << mpi_rank_ <<
    " size " << mpi_size_ << std::endl;

  if (args_.forced_master_is_worker) {
    master_is_worker_ = true;
  } else {
    master_is_worker_ = (mpi_size_ == 1);
  }

  t_load_network_.start();
  MasterAwareLoadNetwork();
  t_load_network_.stop();

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

  sampler_stochastic_info(std::cerr);

  InitDKVStore();

  // Need to know max_perplexity_chunk_ to Init perp_
  perp_.Init(max_perplexity_chunk_);

  init_theta();

  t_populate_pi_.start();
  init_pi();
  t_populate_pi_.stop();

  pi_update_.resize(max_dkv_write_entries_);
  for (auto &p : pi_update_) {
    p = new Float[K + 1];
  }
  phi_node_.resize(max_dkv_write_entries_);
  for (auto &p : phi_node_) {
    p.resize(K + 1);
  }
  grads_beta_.resize(omp_get_max_threads());
  for (auto &g : grads_beta_) {
    // gradients K*2 dimension
    g = std::vector<std::vector<Float> >(2, std::vector<Float>(K));
  }

  chunk_pipeline_ = std::unique_ptr<ChunkPipeline>(
                      new ChunkPipeline(num_buffers_, *d_kv_store_));
  dkv_server_thread_ = boost::thread(boost::ref(*chunk_pipeline_.get()));
  minibatch_pipeline_ = std::unique_ptr<MinibatchPipeline>(
                          new MinibatchPipeline(*this, max_minibatch_chunk_,
                                                *chunk_pipeline_, rng_,
                                                args_.REPLICATED_NETWORK));
}


std::ostream& MCMCSamplerStochasticDistributed::PrintStats(
    std::ostream& out) const {
  Timer::printHeader(out);
  out << t_load_network_ << std::endl;
  out << t_init_dkv_ << std::endl;
  out << t_populate_pi_ << std::endl;
  out << t_outer_ << std::endl;
  out << t_deploy_minibatch_ << std::endl;
  out << t_scatter_subgraph_ << std::endl;
  out << t_scatter_subgraph_marshall_edge_count_ << std::endl;
  out << t_scatter_subgraph_scatterv_edge_count_ << std::endl;
  out << t_scatter_subgraph_marshall_edges_ << std::endl;
  out << t_scatter_subgraph_scatterv_edges_ << std::endl;
  out << t_scatter_subgraph_unmarshall_ << std::endl;
  out << t_mini_batch_ << std::endl;
  out << t_nodes_in_mini_batch_ << std::endl;
  out << t_broadcast_theta_beta_ << std::endl;
  out << t_update_phi_pi_ << std::endl;
  minibatch_pipeline_->report(std::cout);
  chunk_pipeline_->report(std::cout);
  out << t_update_phi_ << std::endl;
  out << t_barrier_phi_ << std::endl;
  out << t_update_pi_ << std::endl;
  out << t_store_pi_minibatch_ << std::endl;
  out << t_barrier_pi_ << std::endl;
  out << t_update_beta_ << std::endl;
  out << t_beta_zero_ << std::endl;
  out << t_beta_rank_ << std::endl;
  out << t_load_pi_beta_ << std::endl;
  out << t_beta_calc_grads_ << std::endl;
  out << t_beta_sum_grads_ << std::endl;
  out << t_beta_reduce_grads_ << std::endl;
  out << t_beta_update_theta_ << std::endl;
  out << t_perplexity_ << std::endl;
  out << t_load_pi_perp_ << std::endl;
  out << t_cal_edge_likelihood_ << std::endl;
  out << t_purge_pi_perp_ << std::endl;
  out << t_reduce_perp_ << std::endl;

  return out;
}


void MCMCSamplerStochasticDistributed::save_pi(::size_t step_count) {
  std::ofstream save;
  boost::filesystem::path dir(args_.dump_pi_file_);
  boost::filesystem::create_directories(dir.parent_path());
  const ::size_t chunk_size = max_dkv_pi_cache_;
  std::vector<Float*> pi(chunk_size);
  ::size_t stored = 0;
  std::string filename = args_.dump_pi_file_;
  if (step_count != 0) {
    filename = filename + "." + std::to_string(step_count);
  }
  std::cerr << "Save pi to file " << filename << std::endl;
  save.open(filename, std::ios::out | std::ios::binary);
  std::cerr << "mpi rank " << mpi_rank_ << " size " << mpi_size_ << std::endl;
  save.write(reinterpret_cast<char *>(&N), sizeof N);
  save.write(reinterpret_cast<char *>(&K), sizeof K);
  int32_t hosts_pi = master_hosts_pi_;
  save.write(reinterpret_cast<char *>(&hosts_pi), sizeof hosts_pi);
  save.write(reinterpret_cast<char *>(&mpi_size_), sizeof mpi_size_);
  save.write(reinterpret_cast<char *>(&mpi_rank_), sizeof mpi_rank_);
  // padding
  for (auto i = 0; i < 3; i++) {
    save.write(reinterpret_cast<char *>(&i), sizeof i);
  }
  while (stored < N) {
    ::size_t chunk = std::min(chunk_size, N - stored);
    std::vector<int> nodes(chunk);
    for (::size_t i = 0; i < chunk; ++i) {
      nodes[i] = stored + i;
    }
    d_kv_store_->ReadKVRecords(0, pi, nodes);
    for (::size_t i = 0; i < chunk; ++i) {
      save.write(reinterpret_cast<char *>(&stored), sizeof stored);
      save.write(reinterpret_cast<char *>(pi[i]), (K + 1)* sizeof pi[i][0]);
      stored++;
    }
    d_kv_store_->PurgeKVRecords(0);
  }
  save.close();
  std::cerr << "Saved pi to file " << filename << std::endl;
}


void MCMCSamplerStochasticDistributed::run() {
  /** run mini-batch based MCMC sampler, based on the sungjin's note */

  using namespace std::chrono;

  PRINT_MEM_USAGE();

  int r;

  r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Barrier(initial) fails");

  t_start_ = std::chrono::system_clock::now();

  minibatch_pipeline_->StageNextMinibatch();

  while (step_count < max_iteration && ! is_converged()) {

    if (args_.dump_pi_file_ != "" && (step_count - 1) % args_.dump_pi_interval_ == 0) {
      save_pi(step_count);
    }
    t_outer_.start();
    // auto l1 = std::chrono::system_clock::now();
    //if (step_count > 200000){
    //interval = 2;
    //}

    minibatch_pipeline_->AdvanceMinibatch();

    broadcast_theta_beta();

    // requires beta at the workers
    check_perplexity(false);
    check_dynamic_step();

    t_update_phi_pi_.start();
    update_phi(&phi_node_);

    // barrier: peers must not read pi already updated in the current iteration
    t_barrier_phi_.start();
    r = MPI_Barrier(MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Barrier(post pi) fails");
    t_barrier_phi_.stop();

    const MinibatchSlice& mb_slice = minibatch_pipeline_->CurrentSlice();
    update_pi(mb_slice, phi_node_);

    // barrier: ensure we read pi/phi_sum from current iteration
    t_barrier_pi_.start();
    r = MPI_Barrier(MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Barrier(post pi) fails");
    t_barrier_pi_.stop();
    t_update_phi_pi_.stop();

    t_update_beta_.start();
    update_beta(*mb_slice.edge_sample_.first, mb_slice.edge_sample_.second);
    t_update_beta_.stop();

    if (mpi_rank_ == mpi_master_) {
      delete mb_slice.edge_sample_.first;
    }

    ++step_count;
    t_outer_.stop();
    // auto l2 = std::chrono::system_clock::now();

    if (step_count % stats_print_interval_ == 0) {
      PrintStats(std::cout);
    }
  }

  r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Barrier(post pi) fails");

  check_perplexity(true);

  chunk_pipeline_->Stop();
  dkv_server_thread_.join();

  r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Barrier(post pi) fails");

  std::cout << "*************** converged *******************" << std::endl;
  PrintStats(std::cout);

  if (args_.dump_pi_file_ != "") {
    save_pi();
  }
}


::size_t MCMCSamplerStochasticDistributed::real_num_node_sample() const {
  return num_node_sample + 1;
}


void MCMCSamplerStochasticDistributed::init_theta() {
  if (mpi_rank_ == mpi_master_) {
    // model parameters and re-parameterization
    // since the model parameter - \pi and \beta should stay in the simplex,
    // we need to restrict the sum of probability equals to 1.  The way we
    // restrict this is using re-reparameterization techniques, where we
    // introduce another set of variables, and update them first followed by
    // updating \pi and \beta.
    // parameterization for \beta
    theta = rng_[0]->gamma(eta[0], eta[1], K, 2);
  } else {
    theta = std::vector<std::vector<Float> >(K, std::vector<Float>(2));
  }
  // std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
  // parameterization for \beta
  // theta = rng_[0]->->gamma(100.0, 0.01, K, 2);
}


void MCMCSamplerStochasticDistributed::beta_from_theta() {
  std::vector<std::vector<Float> > temp(theta.size(),
                                         std::vector<Float>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<Float>(1));
#ifndef NDEBUG
  for (::size_t k = 0; k < K; ++k) {
    assert(! isnan(beta[k]));
  }
#endif
}


// Calculate pi[0..K> ++ phi_sum from phi[0..K>
void MCMCSamplerStochasticDistributed::pi_from_phi(
    Float* pi, const std::vector<Float> &phi) {
  Float phi_sum = std::accumulate(phi.begin(), phi.begin() + K, 0.0);
  for (::size_t k = 0; k < K; ++k) {
    pi[k] = phi[k] / phi_sum;
  }

  pi[K] = phi_sum;
}


void MCMCSamplerStochasticDistributed::init_pi() {
  std::vector<Float*> pi(max_dkv_write_entries_);
  for (auto & p : pi) {
    p = new Float[K + 1];
  }

  ::size_t servers = master_hosts_pi_ ? mpi_size_ : mpi_size_ - 1;
  ::size_t my_max = N / servers;
  int my_server = master_hosts_pi_ ? mpi_rank_ : mpi_rank_ - 1;
  if (my_server < 0) {
    my_max = 0;
  } else if (static_cast<::size_t>(my_server) < N - my_max * servers) {
    ++my_max;
  }
  int last_node = my_server;
  while (my_max > 0) {
    ::size_t chunk = std::min(max_dkv_write_entries_, my_max);
    my_max -= chunk;
    std::vector<std::vector<Float> > phi_pi(chunk);
#pragma omp parallel for // num_threads (12)
    for (::size_t j = 0; j < chunk; ++j) {
      phi_pi[j] = rng_[omp_get_thread_num()]->gamma(1, 1, 1, K)[0];
    }
#ifndef NDEBUG
    for (auto & phs : phi_pi) {
      for (auto ph : phs) {
        assert(ph >= 0.0);
      }
    }
#endif

#pragma omp parallel for // num_threads (12)
    for (::size_t j = 0; j < chunk; ++j) {
      pi_from_phi(pi[j], phi_pi[j]);
    }

    std::vector<int32_t> node(chunk);
    for (::size_t j = 0; j < chunk; ++j) {
      node[j] = last_node;
      last_node += servers;
    }

    d_kv_store_->WriteKVRecords(node, constify(pi));
    d_kv_store_->FlushKVRecords();
    std::cerr << ".";
  }
  std::cerr << std::endl;

  for (auto & p : pi) {
    delete[] p;
  }
}

void MCMCSamplerStochasticDistributed::pi_stats(PiStats *stats) {
  std::vector<Float*> pi(max_dkv_pi_cache_);
  ::size_t servers = master_hosts_pi_ ? mpi_size_ : mpi_size_ - 1;
  ::size_t my_max = N / servers;
  int my_server = master_hosts_pi_ ? mpi_rank_ : mpi_rank_ - 1;
  if (my_server < 0) {
    my_max = 0;
  } else if (static_cast<::size_t>(my_server) < N - my_max * servers) {
    ++my_max;
  }
  ::size_t my_pi_elts = my_max;
  int last_node = my_server;
  ::size_t thread_count = omp_get_max_threads();
  // ::size_t thread_count = max_dkv_pi_cache_;
  std::vector<double> sum = std::vector<double>(thread_count);
  std::vector<double> sumsq = std::vector<double>(thread_count);
  double check_sum = 0.0;
  if (sum[0] != 0.0) {
    std::cerr << "Ooppss... this vector elt should be 0" << std::endl;
  }
  while (my_max > 0) {
    ::size_t chunk = std::min(max_dkv_pi_cache_, my_max);
    my_max -= chunk;

    std::vector<int32_t> node(chunk);
    for (::size_t j = 0; j < chunk; ++j) {
      node[j] = last_node;
      last_node += servers;
    }

    d_kv_store_->ReadKVRecords(0, pi, node);

// #pragma omp parallel for // num_threads (12)
    for (::size_t j = 0; j < chunk; ++j) {
      double sum_j = 0.0;
      double sumsq_j = 0.0;
      for (::size_t k = 0; k < K; ++k) {
        check_sum += pi[j][k];
        sum_j += pi[j][k];
        sumsq_j += pi[j][k] * pi[j][k];
      }
      sum[omp_get_thread_num()] += sum_j;
      sumsq[omp_get_thread_num()] += sumsq_j;
    }

    d_kv_store_->PurgeKVRecords(0);
  }

  // Do a local reduce(+)
  // Abuse the existing accu (which was tuned for perplexity)
  perp_.accu_[0].link.count = my_pi_elts * K;       // N
  perp_.accu_[0].link.likelihood = 0.0;         // sum
  perp_.accu_[0].non_link.likelihood = 0.0;     // sumsq
  for (::size_t j = 0; j < sum.size(); ++j) {
    perp_.accu_[0].link.likelihood += (Float)sum[j];
    perp_.accu_[0].non_link.likelihood += (Float)sumsq[j];
  }
  std::cerr.setf(std::ios_base::fmtflags(0), std::ios_base::floatfield);
  std::cerr << std::setprecision(6);

  // Do a distributed reduce(+)
  perp_accu accu;
  t_reduce_perp_.start();
  reduce_plus(perp_.accu_[0], &accu);
  t_reduce_perp_.stop();

  std::cerr << mpi_rank_ << ": N " << perp_.accu_[0].link.count << " sum " << perp_.accu_[0].link.likelihood << " (check " << check_sum << " sum[0] " << sum[0] << ") sumsq " << perp_.accu_[0].non_link.likelihood << std::endl;

  auto global_sum = accu.link.likelihood;
  auto global_sumsq = accu.non_link.likelihood;
  auto N = accu.link.count;
  stats->N_ = N;
  stats->mean_ = global_sum / N;
  stats->stdev_ = stdev(global_sum, global_sumsq, N);
}


void MCMCSamplerStochasticDistributed::check_perplexity(bool force) {
  if (force || (step_count - 1) % interval == 0) {
    using namespace std::chrono;

    t_perplexity_.start();
    // TODO load pi for the held-out set to calculate perplexity
    Float ppx_score = cal_perplexity_held_out();
    t_perplexity_.stop();
    PiStats psts;
    if (args_.pi_stats_) {
      pi_stats(&psts);
    }
    if (mpi_rank_ == mpi_master_) {
      auto t_now = system_clock::now();
      auto t_ms = duration_cast<milliseconds>(t_now - t_start_).count();
      std::cout << "average_count is: " << average_count << " ";
      std::cout << std::fixed
                << "step count: " << step_count
                << " time: " << std::setprecision(3) << (t_ms / 1000.0)
                << " perplexity for hold out set: " << std::setprecision(12) <<
                ppx_score << std::endl;
      double seconds = t_ms / 1000.0;
      timings_.push_back(seconds);
      if (args_.pi_stats_) {
        std::cout << "Pi N " << psts.N_ << " average "
                  << psts.mean_ << " stdev " << psts.stdev_ << std::endl;
      }
    }

    ppxs_heldout_cb_.push_back(ppx_score);
  }
}


void MCMCSamplerStochasticDistributed::ScatterSubGraph(
    const std::vector<std::vector<Vertex> > &subminibatch,
    MinibatchSlice *mb_slice) {
  std::vector<int32_t> set_size(mb_slice->nodes_.size());
  std::vector<Vertex> flat_subgraph;
  int r;

  mb_slice->local_network_.reset();

  if (mpi_rank_ == mpi_master_) {
    std::vector<int32_t> size_count(mpi_size_);	        // FIXME: lift to class
    std::vector<int32_t> size_displ(mpi_size_);	        // FIXME: lift to class
    std::vector<int32_t> subgraph_count(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> subgraph_displ(mpi_size_);	// FIXME: lift to class
    std::vector<int32_t> workers_set_size;

    // Data dependency on workers_set_size
    t_scatter_subgraph_marshall_edge_count_.start();
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
    t_scatter_subgraph_marshall_edge_count_.stop();

    // Scatter the fanout of each minibatch node
    t_scatter_subgraph_scatterv_edge_count_.start();
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
    t_scatter_subgraph_scatterv_edge_count_.stop();

    // Marshall the subgraphs
    t_scatter_subgraph_marshall_edges_.start();
    ::size_t total_edges = np::sum(workers_set_size);
    std::vector<Vertex> subgraphs(total_edges);
#pragma omp parallel for // num_threads (12)
    for (int i = 0; i < mpi_size_; ++i) {
      ::size_t marshalled = subgraph_displ[i];
      for (::size_t j = 0; j < subminibatch[i].size(); ++j) {
        Vertex* marshall = subgraphs.data() + marshalled;
        ::size_t n = network.marshall_edges_from(subminibatch[i][j],
                                                 marshall);
        // std::cerr << "Marshall to peer " << i << ": " << n <<
        //   " edges" << std::endl;
        marshalled += n;
      }
    }
    t_scatter_subgraph_marshall_edges_.stop();

    // Scatter the marshalled subgraphs
    t_scatter_subgraph_scatterv_edges_.start();
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
    t_scatter_subgraph_scatterv_edges_.stop();

  } else {
    // Scatter the fanout of each minibatch node
    t_scatter_subgraph_scatterv_edge_count_.start();
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
    t_scatter_subgraph_scatterv_edge_count_.stop();

    // Scatter the marshalled subgraphs
    t_scatter_subgraph_scatterv_edges_.start();
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
    t_scatter_subgraph_scatterv_edges_.stop();
  }

  t_scatter_subgraph_unmarshall_.start();
  ::size_t offset = 0;
  for (::size_t i = 0; i < set_size.size(); ++i) {
    Vertex* marshall = &flat_subgraph[offset];
    mb_slice->local_network_.unmarshall_local_graph(i, marshall, set_size[i]);
    offset += set_size[i];
  }
  t_scatter_subgraph_unmarshall_.stop();
}


void MCMCSamplerStochasticDistributed::deploy_mini_batch(
    MinibatchSlice* mb_slice) {
  t_deploy_minibatch_.start();
  std::vector<std::vector<int> > subminibatch;
  std::vector<int32_t> minibatch_chunk(mpi_size_);      // FIXME: lift to class
  std::vector<int32_t> scatter_minibatch;               // FIXME: lift to class
  std::vector<int32_t> scatter_displs(mpi_size_);       // FIXME: lift to class
  int		r;
  std::vector<Vertex> nodes_vector; // FIXME: duplicates scatter_minibatch

  if (mpi_rank_ == mpi_master_) {
    // std::cerr << "Invoke sample_mini_batch" << std::endl;
    t_mini_batch_.start();
    mb_slice->edge_sample_ = network.sample_mini_batch(mini_batch_size,
                                                       strategy);
    t_mini_batch_.stop();
    const MinibatchSet &mini_batch = *mb_slice->edge_sample_.first;
    // std::cerr << "Done sample_mini_batch" << std::endl;

    t_nodes_in_mini_batch_.start();
    MinibatchNodeSet nodes = nodes_in_batch(mini_batch);
    nodes_vector.assign(nodes.begin(), nodes.end());
    t_nodes_in_mini_batch_.stop();
    // std::cerr << "mini_batch size " << mini_batch.size() <<
    //   " num_node_sample " << num_node_sample << std::endl;

    subminibatch.resize(mpi_size_);	// FIXME: lift to class, size is static

    ::size_t workers = master_is_worker_ ? mpi_size_ : mpi_size_ - 1;
    ::size_t upper_bound = (nodes_vector.size() + workers - 1) / workers;
    std::unordered_set<Vertex> unassigned;
    for (auto n: nodes_vector) {
      ::size_t owner = node_owner(n);
      if (subminibatch[owner].size() == upper_bound) {
        unassigned.insert(n);
      } else {
        subminibatch[owner].push_back(n);
      }
    }

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
  mb_slice->nodes_.resize(my_minibatch_size);
  if (mb_slice->nodes_.size() > pi_update_.size()) {
    PRINT_MEM_USAGE();
    std::ostringstream msg;
    msg << "Out of bounds for pi_update_/phi_node_: bounds " << pi_update_.size() << " required " << mb_slice->nodes_.size();
    throw BufferSizeException(msg.str());
  }

  if (mpi_rank_ == mpi_master_) {
    // TODO Master scatters the minibatch nodes over the workers,
    // preferably with consideration for both load balance and locality
    r = MPI_Scatterv(scatter_minibatch.data(),
                     minibatch_chunk.data(),
                     scatter_displs.data(),
                     MPI_INT,
                     mb_slice->nodes_.data(),
                     my_minibatch_size,
                     MPI_INT,
                     mpi_master_,
                     MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");

  } else {
    r = MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     mb_slice->nodes_.data(), my_minibatch_size, MPI_INT,
                     mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch fails");
  }

  if (! args_.REPLICATED_NETWORK) {
    t_scatter_subgraph_.start();
    ScatterSubGraph(subminibatch, mb_slice);
    t_scatter_subgraph_.stop();
  }

  // Broadcast nodes of the full minibatch
  int32_t num_nodes = nodes_vector.size();
  r = MPI_Bcast(&num_nodes, 1, MPI_INT, mpi_master_, MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Bcast of minibatch num_nodes fails");
  nodes_vector.resize(num_nodes);
  r = MPI_Bcast(nodes_vector.data(), num_nodes, MPI_INT, mpi_master_,
                MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Bcast of minibatch nodes fails");
  mb_slice->full_minibatch_nodes_.clear();
  mb_slice->full_minibatch_nodes_.insert(nodes_vector.begin(),
                                         nodes_vector.end());
  t_deploy_minibatch_.stop();

  // std::cerr << step_count << ": Minibatch deployed" << std::endl;
}


void MCMCSamplerStochasticDistributed::update_phi(
    std::vector<std::vector<Float> >* phi_node) {
  Float eps_t = get_eps_t();

  assert(minibatch_pipeline_->CurrentSlice().pi_chunks_.size() >= 2);
  for (::size_t b = 0;
       b < minibatch_pipeline_->CurrentSlice().pi_chunks_.size();
       ++b) {
    minibatch_pipeline_->StageNextChunk();                   // next chunk
    auto *pi_chunk = chunk_pipeline_->AwaitChunkFilled();   // current chunk

    // std::cerr << step_count << ": compute chunk " << b << " size " << pi_chunk->chunk_nodes_.size() << std::endl;
    t_update_phi_.start();
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < pi_chunk->chunk_nodes_.size(); ++i) {
      Vertex node = pi_chunk->chunk_nodes_[i];
      update_phi_node(minibatch_pipeline_->CurrentSlice(), *pi_chunk,
                      pi_chunk->start_ + i, node, pi_chunk->pi_node_[i],
                      eps_t, rng_[omp_get_thread_num()],
                      &(*phi_node)[pi_chunk->start_ + i]);
    }
    t_update_phi_.stop();

    chunk_pipeline_->ReleaseChunk();
    // d_kv_store_->PurgeKVRecords(pi_chunk->buffer_index_);
  }
}


void MCMCSamplerStochasticDistributed::update_phi_node(
    const MinibatchSlice& mb_slice, const PiChunk& pi_chunk,
    ::size_t index, Vertex i, const Float* pi_node,
    Float eps_t, Random::Random* rnd,
    std::vector<Float>* phi_node	// out parameter
    ) {
  Float phi_i_sum = pi_node[K];
  if (phi_i_sum == FLOAT(0.0)) {
    std::cerr << "Ooopppssss.... phi_i_sum " << phi_i_sum << std::endl;
  }
  std::vector<Float> grads(K, 0.0);	// gradient for K classes

  for (::size_t ix = 0; ix < real_num_node_sample(); ++ix) {
    Vertex neighbor = pi_chunk.flat_neighbors_[ix];
    if (i != neighbor) {
      int y_ab = 0;		// observation
      if (args_.REPLICATED_NETWORK) {
        Edge edge(std::min(i, neighbor), std::max(i, neighbor));
        if (edge.in(network.get_linked_edges())) {
          y_ab = 1;
        }
      } else {
        Edge edge(index, neighbor);
        if (mb_slice.local_network_.find(edge)) {
          y_ab = 1;
        }
      }

      std::vector<Float> probs(K);
      Float e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
      for (::size_t k = 0; k < K; ++k) {
        Float f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
        probs[k] = pi_node[k] * (pi_chunk.pi_neighbor_[ix][k] * f + e);
        assert(! isnan(probs[k]));
      }

      Float prob_sum = np::sum(probs);
      assert(prob_sum > 0);
      // std::cerr << std::fixed << std::setprecision(12) << "node " << i <<
      //    " neighb " << neighbor << " prob_sum " << prob_sum <<
      //    " phi_i_sum " << phi_i_sum <<
      //    " #sample " << real_num_node_sample() << std::endl;
      for (::size_t k = 0; k < K; ++k) {
        assert(phi_i_sum > 0);
        grads[k] += ((probs[k] / prob_sum) / pi_node[k] - 1.0) / phi_i_sum;
        assert(! isnan(grads[k]));
      }
    } else {
      // std::cerr << "Skip self loop <" << i << "," << neighbor << ">" << std::endl;
    }
  }

  std::vector<Float> noise = rnd->randn(K);	// random gaussian noise.
  Float Nn = (1.0 * N) / num_node_sample;
  // update phi for node i
  for (::size_t k = 0; k < K; ++k) {
    Float phi_node_k = pi_node[k] * phi_i_sum;
    assert(phi_node_k > FLOAT(0.0));
    phi_node_k = std::abs(phi_node_k + eps_t / 2 * (alpha - phi_node_k +
                                                    Nn * grads[k]) +
                          sqrt(eps_t * phi_node_k) * noise[k]);
    if (phi_node_k < MCMC_NONZERO_GUARD) {
      (*phi_node)[k] = MCMC_NONZERO_GUARD;
    } else {
      (*phi_node)[k] = phi_node_k;
    }
    assert((*phi_node)[k] > FLOAT(0.0));
  }
}


void MCMCSamplerStochasticDistributed::update_pi(
    const MinibatchSlice& mb_slice,
    const std::vector<std::vector<Float> >& phi_node) {
  // calculate and store updated values for pi/phi_sum

  if (mpi_rank_ != mpi_master_ || master_is_worker_) {
    t_update_pi_.start();
#pragma omp parallel for // num_threads (12)
    for (::size_t i = 0; i < mb_slice.nodes_.size(); ++i) {
      pi_from_phi(pi_update_[i], phi_node[i]);
    }
    t_update_pi_.stop();

    t_store_pi_minibatch_.start();
    d_kv_store_->WriteKVRecords(mb_slice.nodes_, constify(pi_update_));
    t_store_pi_minibatch_.stop();
    d_kv_store_->FlushKVRecords();
  }
}


void MCMCSamplerStochasticDistributed::broadcast_theta_beta() {
  t_broadcast_theta_beta_.start();
  std::vector<Float> theta_marshalled(2 * K);   // FIXME: lift to class level
  if (mpi_rank_ == mpi_master_) {
    for (::size_t k = 0; k < K; ++k) {
      for (::size_t i = 0; i < 2; ++i) {
        theta_marshalled[2 * k + i] = theta[k][i];
      }
    }
  }
  int r = MPI_Bcast(theta_marshalled.data(), theta_marshalled.size(),
                    FLOATTYPE_MPI, mpi_master_, MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Bcast(theta) fails");
  if (mpi_rank_ != mpi_master_) {
    for (::size_t k = 0; k < K; ++k) {
      for (::size_t i = 0; i < 2; ++i) {
        theta[k][i] = theta_marshalled[2 * k + i];
      }
    }
  }
  //-------- after broadcast of theta, replicate this at all peers:
  beta_from_theta();
  t_broadcast_theta_beta_.stop();
}


void MCMCSamplerStochasticDistributed::scatter_minibatch_for_theta(
    const MinibatchSet &mini_batch,
    std::vector<EdgeMapItem>* mini_batch_slice) {
  int   r;
  std::vector<unsigned char> flattened_minibatch;
  std::vector<int32_t> scatter_size(mpi_size_);
  std::vector<int32_t> scatter_displs(mpi_size_);

  if (mpi_rank_ == mpi_master_) {
    flattened_minibatch.resize(mini_batch.size() * sizeof(EdgeMapItem));
    ::size_t chunk = mini_batch.size() / mpi_size_;
    ::size_t surplus = mini_batch.size() - chunk * mpi_size_;
    ::size_t running_sum = 0;
    ::size_t i;
    for (i = 0; i < surplus; ++i) {
      scatter_size[i] = (chunk + 1) * sizeof(EdgeMapItem);
      scatter_displs[i] = running_sum;
      running_sum += (chunk + 1) * sizeof(EdgeMapItem);
    }
    for (; i < (::size_t)mpi_size_; ++i) {
      scatter_size[i] = chunk * sizeof(EdgeMapItem);
      scatter_displs[i] = running_sum;
      running_sum += chunk * sizeof(EdgeMapItem);
    }
    auto *marshall = flattened_minibatch.data();
    for (auto e: mini_batch) {
      EdgeMapItem ei(e, e.in(network.get_linked_edges()));
      memcpy(marshall, &ei, sizeof ei);
      marshall += sizeof ei;
    }
  }

  int32_t my_minibatch_bytes;
  r = MPI_Scatter(scatter_size.data(), 1, MPI_INT,
                  &my_minibatch_bytes, 1, MPI_INT,
                  mpi_master_, MPI_COMM_WORLD);
  mpi_error_test(r, "MPI_Scatter of minibatch sizes for update_beta fails");

  mini_batch_slice->resize(my_minibatch_bytes / sizeof(EdgeMapItem));
  if (mpi_rank_ == mpi_master_) {
    r = MPI_Scatterv(flattened_minibatch.data(), scatter_size.data(),
                     scatter_displs.data(), MPI_BYTE,
                     mini_batch_slice->data(), my_minibatch_bytes, MPI_BYTE,
                     mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch for update_beta fails");

  } else {
    r = MPI_Scatterv(NULL, NULL,
                     NULL, MPI_BYTE,
                     mini_batch_slice->data(), my_minibatch_bytes, MPI_BYTE,
                     mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "MPI_Scatterv of minibatch for update_beta fails");
  }
}


void MCMCSamplerStochasticDistributed::beta_calc_grads(
    const std::vector<EdgeMapItem>& mini_batch_slice) {
  t_beta_zero_.start();
#pragma omp parallel for
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    for (::size_t k = 0; k < K; ++k) {
      grads_beta_[i][0][k] = 0.0;
      grads_beta_[i][1][k] = 0.0;
    }
  }

  // sums = np.sum(self.__theta,1)
  std::vector<Float> theta_sum(theta.size());
  std::transform(theta.begin(), theta.end(), theta_sum.begin(),
                 np::sum<Float>);
  t_beta_zero_.stop();

  t_beta_rank_.start();
  std::unordered_map<Vertex, Vertex> node_rank;
  std::vector<Vertex> nodes;
  for (auto e : mini_batch_slice) {
    Vertex i = e.edge.first;
    Vertex j = e.edge.second;
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

  std::vector<Float*> pi(node_rank.size());
  t_load_pi_beta_.start();
  ::size_t buffer_index = chunk_pipeline_->GrabFreeBufferIndex();
  d_kv_store_->ReadKVRecords(buffer_index, pi, nodes);
  t_load_pi_beta_.stop();

  // update gamma, only update node in the grad
  t_beta_calc_grads_.start();
#pragma omp parallel for // num_threads (12)
  for (::size_t e = 0; e < mini_batch_slice.size(); ++e) {
    const auto *edge = &mini_batch_slice[e];
    std::vector<Float> probs(K);

    int y = (int)edge->is_edge;
    Vertex i = node_rank[edge->edge.first];
    Vertex j = node_rank[edge->edge.second];

    Float pi_sum = 0.0;
    for (::size_t k = 0; k < K; ++k) {
      // Note: this is the KV-store cached pi, not the Learner item
      Float f = pi[i][k] * pi[j][k];
      pi_sum += f;
      if (y == 1) {
        probs[k] = beta[k] * f;
      } else {
        probs[k] = (1.0 - beta[k]) * f;
      }
    }

    Float prob_0 = ((y == 1) ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
    Float prob_sum = np::sum(probs) + prob_0;
    assert(prob_sum > FLOAT(0.0));
    for (::size_t k = 0; k < K; ++k) {
      Float f = probs[k] / prob_sum;
      Float one_over_theta_sum = 1.0 / theta_sum[k];

      grads_beta_[omp_get_thread_num()][0][k] += f * ((1 - y) / theta[k][0] -
                                                      one_over_theta_sum);
      grads_beta_[omp_get_thread_num()][1][k] += f * (y / theta[k][1] -
                                                      one_over_theta_sum);
    }
  }

  chunk_pipeline_->ReleaseGrabbedBuffer(buffer_index);
  t_beta_calc_grads_.stop();
}


void MCMCSamplerStochasticDistributed::beta_sum_grads() {
  int r;

  t_beta_sum_grads_.start();
#pragma omp parallel for
  for (::size_t k = 0; k < K; ++k) {
    for (int i = 1; i < omp_get_max_threads(); ++i) {
      grads_beta_[0][0][k] += grads_beta_[i][0][k];
      grads_beta_[0][1][k] += grads_beta_[i][1][k];
    }
  }
  t_beta_sum_grads_.stop();

  //-------- reduce(+) of the grads_[0][*][0,1] to the master
  t_beta_reduce_grads_.start();
  if (mpi_rank_ == mpi_master_) {
    r = MPI_Reduce(MPI_IN_PLACE, grads_beta_[0][0].data(), K, FLOATTYPE_MPI,
                   MPI_SUM, mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "Reduce/plus of grads_beta_[0][0] fails");
    r = MPI_Reduce(MPI_IN_PLACE, grads_beta_[0][1].data(), K, FLOATTYPE_MPI,
                   MPI_SUM, mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "Reduce/plus of grads_beta_[0][1] fails");
  } else {
    r = MPI_Reduce(grads_beta_[0][0].data(), NULL, K, FLOATTYPE_MPI,
                   MPI_SUM, mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "Reduce/plus of grads_beta_[0][0] fails");
    r = MPI_Reduce(grads_beta_[0][1].data(), NULL, K, FLOATTYPE_MPI,
                   MPI_SUM, mpi_master_, MPI_COMM_WORLD);
    mpi_error_test(r, "Reduce/plus of grads_beta_[0][1] fails");
  }
  t_beta_reduce_grads_.stop();
}


void MCMCSamplerStochasticDistributed::beta_update_theta(Float scale) {
  if (mpi_rank_ == mpi_master_) {
    t_beta_update_theta_.start();
    Float eps_t = get_eps_t();
    // random noise.
    std::vector<std::vector<Float> > noise = rng_[0]->randn(K, 2);
#pragma omp parallel for
    for (::size_t k = 0; k < K; ++k) {
      for (::size_t i = 0; i < 2; ++i) {
        Float f = std::sqrt(eps_t * theta[k][i]);
        theta[k][i] = std::abs(theta[k][i] +
                               eps_t / 2.0 * (eta[i] - theta[k][i] +
                                              scale * grads_beta_[0][i][k]) +
                               f * noise[k][i]);
        if (theta[k][i] < MCMC_NONZERO_GUARD) {
          theta[k][i] = MCMC_NONZERO_GUARD;
        }
      }
    }
    t_beta_update_theta_.stop();
  }
}


void MCMCSamplerStochasticDistributed::update_beta(
    const MinibatchSet &mini_batch, Float scale) {
  std::vector<EdgeMapItem> mini_batch_slice;

  scatter_minibatch_for_theta(mini_batch, &mini_batch_slice);

  beta_calc_grads(mini_batch_slice);

  beta_sum_grads();

  beta_update_theta(scale);
}


void MCMCSamplerStochasticDistributed::reduce_plus(const perp_accu &in,
                                                   perp_accu* accu) {
  int r;
  uint64_t count[2] = { in.link.count, in.non_link.count };
  Float likelihood[2] = { in.link.likelihood, in.non_link.likelihood };

  r = MPI_Allreduce(MPI_IN_PLACE, count, 2, MPI_UNSIGNED_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
  mpi_error_test(r, "Reduce/plus of perplexity counts fails");
  r = MPI_Allreduce(MPI_IN_PLACE, likelihood, 2, FLOATTYPE_MPI, MPI_SUM,
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
Float MCMCSamplerStochasticDistributed::cal_perplexity_held_out() {
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
    std::vector<Vertex> chunk_nodes(perp_.nodes_.begin() + 2 * chunk_start,
                                    perp_.nodes_.begin() + 2 * (chunk_start +
                                                                chunk));

    t_load_pi_perp_.start();
    ::size_t buffer_index = chunk_pipeline_->GrabFreeBufferIndex();
    d_kv_store_->ReadKVRecords(buffer_index, perp_.pi_, chunk_nodes);
    t_load_pi_perp_.stop();

    t_cal_edge_likelihood_.start();
#pragma omp parallel for
    for (::size_t i = chunk_start; i < chunk_start + chunk; ++i) {
      const auto& edge_in = perp_.data_[i];
      // the index into the nodes/pi vectors is double the index into the
      // edge vector (+ 1)
      Vertex a = 2 * (i - chunk_start);
      Vertex b = 2 * (i - chunk_start) + 1;
      Float edge_likelihood = cal_edge_likelihood(perp_.pi_[a], perp_.pi_[b],
                                                   edge_in.is_edge, beta);
      if (std::isnan(edge_likelihood)) {
        std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
      }

      //cout<<"AVERAGE COUNT: " <<average_count;
      ppx_per_heldout_edge_[i] = (ppx_per_heldout_edge_[i] * (average_count-1) + edge_likelihood)/(average_count);
      // Edge e(chunk_nodes[a], chunk_nodes[b]);
      // std::cout << std::fixed << std::setprecision(12) << e <<
      //   " in? " << (edge_in.is_edge ? "True" : "False") <<
      //   " -> " << edge_likelihood << " av. " << average_count <<
      //   " ppx[" << i << "] " << ppx_per_heldout_edge_[i] << std::endl;
      if (edge_in.is_edge) {
        perp_.accu_[omp_get_thread_num()].link.count++;
        perp_.accu_[omp_get_thread_num()].link.likelihood += std::log(ppx_per_heldout_edge_[i]);
        //link_likelihood += edge_likelihood;

        if (std::isnan(perp_.accu_[omp_get_thread_num()].link.likelihood)){
          std::cerr << "link_likelihood is NaN; potential bug" << std::endl;
        }
      } else {
        perp_.accu_[omp_get_thread_num()].non_link.count++;
        //perp_.accu_[omp_get_thread_num()].non_link.likelihood +=
        //  edge_likelihood;
        perp_.accu_[omp_get_thread_num()].non_link.likelihood +=
          std::log(ppx_per_heldout_edge_[i]);
        if (std::isnan(perp_.accu_[omp_get_thread_num()].non_link.likelihood)){
          std::cerr << "non_link_likelihood is NaN; potential bug" << std::endl;
        }
      }
    }

    t_purge_pi_perp_.start();
    chunk_pipeline_->ReleaseGrabbedBuffer(buffer_index);
    t_purge_pi_perp_.stop();
  }

  for (auto i = 1; i < omp_get_max_threads(); ++i) {
    perp_.accu_[0].link.count += perp_.accu_[i].link.count;
    perp_.accu_[0].link.likelihood += perp_.accu_[i].link.likelihood;
    perp_.accu_[0].non_link.count += perp_.accu_[i].non_link.count;
    perp_.accu_[0].non_link.likelihood += perp_.accu_[i].non_link.likelihood;
  }

  t_cal_edge_likelihood_.stop();

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
  Float avg_likelihood = 0.0;
  if (accu.link.count + accu.non_link.count != 0){
    avg_likelihood = (accu.link.likelihood + accu.non_link.likelihood) /
      (accu.link.count + accu.non_link.count);
  }

  average_count = average_count + 1;

  return (-avg_likelihood);
}


int MCMCSamplerStochasticDistributed::node_owner(Vertex node) const {
  if (master_hosts_pi_) {
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
