/*
 * Copyright notice
 */

#include "mcmc/config.h"

#include "DKVStoreRDMA.h"

#include <unistd.h>

#ifdef MCMC_ENABLE_RDMA
#include <infiniband/verbs.h>
#endif

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#endif
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif

#include "mcmc/random.h"
#include "mcmc/options.h"


namespace DKV {
namespace DKVRDMA {

/*
 * Class description
 */
DKVStoreRDMA::DKVStoreRDMA() {
}

DKVStoreRDMA::~DKVStoreRDMA() {
  oob_network_.close();

  if (false) {
    std::cerr << "Should linger a bit to allow gracious shutdown" << std::endl;
  } else {
    std::cerr << "Linger a bit to allow gracious shutdown" << std::endl;
    usleep(500000);
  }

  Timer::printHeader(std::cout);
  std::cout << t_poll_cq_ << std::endl;

  std::cout << t_read_.outer << std::endl;
  std::cout << t_read_.local << std::endl;
  std::cout << t_read_.host << std::endl;
  std::cout << t_read_.post << std::endl;
  std::cout << t_read_.finish << std::endl;

  std::cout << t_write_.outer << std::endl;
  std::cout << t_write_.local << std::endl;
  std::cout << t_write_.host << std::endl;
  std::cout << t_write_.post << std::endl;
  std::cout << t_write_.finish << std::endl;

  std::cout << t_barrier_ << std::endl;

  std::cout << "posts " << num_posts_ << " messages " << msgs_per_post_ << " msgs/post " << ((double)msgs_per_post_ / num_posts_) << std::endl;
  // std::chrono::high_resolution_clock::duration dt;
  std::cout << "Local read   " << bytes_local_read     << "B " <<
    GBs_from_timer(t_read_.local, bytes_local_read) << "GB/s" << std::endl;
  auto dt = t_read_.outer.total() - t_read_.local.total();
  auto dth = t_read_.host.total();
  std::cout << "Remote read  " << bytes_remote_read    << "B " <<
    GBs_from_time(dt, bytes_remote_read) << "GB/s per-host loop " <<
    GBs_from_time(dth, bytes_remote_read) << "GB/s" << std::endl;
  std::cout << "Local write  " << bytes_local_written  << "B " <<
    GBs_from_timer(t_write_.local, bytes_local_written) << "GB/s" << std::endl;
  dt = t_write_.outer.total() - t_write_.local.total();
  dth = t_write_.host.total();
  std::cout << "Remote write " << bytes_remote_written << "B " <<
    GBs_from_time(dt, bytes_remote_written) << "GB/s per-host loop " <<
    GBs_from_time(dth, bytes_remote_written) << "GB/s" << std::endl;

  if (rd_close(&res_) != 0) {
    throw QPerfException("rd_close");
  }
  if (rd_close_2(&res_) != 0) {
    throw QPerfException("rd_close_2");
  }
}

double DKVStoreRDMA::GBs_from_time(const double &dt, int64_t bytes) {
  double gbs = bytes / dt / (1LL << 30);

  return gbs;
}

double DKVStoreRDMA::GBs_from_time(
    const std::chrono::high_resolution_clock::duration &dt, int64_t bytes) {
  double s = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count() /
               1000000000.0;

  return GBs_from_time(s, bytes);
}

double DKVStoreRDMA::GBs_from_timer(const Timer &timer, int64_t bytes) {
  return GBs_from_time(timer.total(), bytes);
}

void DKVStoreRDMA::Init(::size_t value_size, ::size_t total_values,
                        ::size_t max_cache_capacity,
                        ::size_t max_write_capacity,
                        const std::vector<std::string> &args) {
  namespace po = ::boost::program_options;

  int ib_port;
  int mtu;

  std::string oob_server;
  uint32_t oob_port;

  bool force_include_master;

  po::options_description desc("RDMA options");
  desc.add_options()
    // ("rdma.oob-port", po::value(&config_.tcp_port)->default_value(0), "RDMA out-of-band TCP port")
    ("rdma.dev", 
     po::value(&dev_name_),
     "RDMA device name")
    ("rdma.port",
     po::value(&ib_port),
     "RDMA device port")
    ("rdma.mtu",
     po::value(&mtu)->default_value(2048),
     "RDMA MTU (256/512/1024/2048/4096) (512)")
    ("rdma.chunk",
     po::value(&post_send_chunk_),
     "RDMA max number of messages per post")
    ("rdma.peers",
     po::value(&batch_size_)->default_value(0),
     "RDMA max number of peers to address in one post")
    ("rdma.include-master",
     po::bool_switch(&force_include_master)->default_value(false),
     "RDMA KV-store includes master node")

    ("rdma.oob-server",
     po::value(&oob_server)->default_value(""),
     "RDMA OOB server")
    ("rdma.oob-port",
     po::value(&oob_port)->default_value(0),
     "RDMA OOB port")
    ("rdma.oob-nhosts",
     po::value(&oob_.num_servers)->default_value(0),
     "RDMA OOB num hosts")
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

  // Feed the options to the QPerf Req
  Req.mtu_size = mtu;
  Req.id = dev_name_.c_str();
  Req.port = ib_port;
  Req.static_rate = "";
  Req.src_path_bits = 0;
  Req.sl = 0;
  Req.poll_mode = 1;
  Req.alt_port = 0;

  Timer::setTabular(true);

  t_poll_cq_      = Timer("RDMA poll cq");
  t_read_.outer   = Timer("RDMA read");
  t_read_.local   = Timer("     local read");
  t_read_.post    = Timer("     post read");
  t_read_.finish  = Timer("     finish read");
  t_read_.host    = Timer("     per-host read");
  t_write_.outer  = Timer("RDMA write");
  t_write_.local  = Timer("     local write");
  t_write_.post   = Timer("     post write");
  t_write_.finish = Timer("     finish write");
  t_write_.host   = Timer("     per-host write");
  t_barrier_      = Timer("RDMA barrier");

  oob_network_.Init(oob_server, oob_port, &oob_.num_servers, &oob_.my_rank);

  if (force_include_master) {
    include_master_ = true;
  } else {
    include_master_ = (oob_.num_servers == 1);
  }
  std::cout << "RDMA D-KV store " << (include_master_ ? "in" : "ex") <<
    "cludes the master node" << std::endl;

  if (rd_open(&res_, IBV_QPT_RC, post_send_chunk_, 0) != 0) {
    throw QPerfException("rd_open");
  }

  if (batch_size_ == 0) {
    if (oob_.num_servers <= 3) {
      batch_size_ = oob_.num_servers;
    } else if (oob_.num_servers <= 6) {
      batch_size_ = 4;
    } else if (oob_.num_servers <= 12) {
      batch_size_ = 5;
    } else if (oob_.num_servers <= 24) {
      batch_size_ = 6;
    } else if (oob_.num_servers <= 48) {
      batch_size_ = 7;
    } else {
      batch_size_ = 8;
    }
  }
  if (batch_size_ > oob_.num_servers) {
    batch_size_ = oob_.num_servers;
  }

  value_size_ = value_size;
  total_values_ = total_values;
  /* memory buffer to hold the value data */
  ::size_t my_values;
  if (include_master_) {
    my_values = (total_values + oob_.num_servers - 1) / oob_.num_servers;
  } else {
    if (oob_.my_rank == 0) {
      // something smallish, does not matter how much
      my_values = max_cache_capacity;
    } else {
      my_values = (total_values + (oob_.num_servers - 1) - 1) / (oob_.num_servers - 1);
    }
  }
  value_.Init(&res_, my_values * value_size);
  std::cout << "MR/value " << value_ << std::endl;

  /* memory buffer to hold the cache data */
  cache_.Init(&res_, &cache_buffer_, max_cache_capacity * value_size);
  std::cout << "MR/cache " << cache_ << std::endl;

  /* memory buffer to hold the zerocopy write data */
  write_.Init(&res_, &write_buffer_, max_write_capacity * value_size);
  std::cout << "MR/write " << write_ << std::endl;

  peer_.resize(oob_.num_servers);
  for (::size_t i = 0; i < oob_.num_servers; i++) {
    if (i == oob_.my_rank) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      peer_[i].Init(&res_);
    }
  }

  /* exchange using o-o-b network info required to connect QPs */
  std::vector<cm_con_data_t> my_con_data(oob_.num_servers);
  for (::size_t i = 0; i < oob_.num_servers; ++i) {
    if (i == oob_.my_rank) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      cm_con_data_t *con = &my_con_data[i];
      con->value      = reinterpret_cast<uint64_t>(value_.area_);
      con->cache      = reinterpret_cast<uint64_t>(cache_.area_);
      con->value_rkey = value_.region_.mr->rkey;
      con->cache_rkey = cache_.region_.mr->rkey;
      con->qp_num     = peer_[i].connection.local.qpn;
      con->psn        = peer_[i].connection.local.psn;
      con->rd_atomic  = peer_[i].connection.local.rd_atomic;
      con->lid        = res_.lnode.lid;
      con->alt_lid    = res_.lnode.alt_lid;
#if 0
      con->gid    = my_gid;
#endif
      std::cout << "My con_data for qp to host " << i << ": " << std::endl;
      std::cout << *con;
      std::cout << "Local LID[" << i << "] = 0x " << std::hex <<
        res_.lnode.lid << std::endl << std::dec;
    }
  }

  std::vector<cm_con_data_t> remote_con_data(oob_.num_servers);
  oob_network_.exchange_oob_data(my_con_data, &remote_con_data);

  for (::size_t i = 0; i < oob_.num_servers; ++i) {
    if (i == oob_.my_rank) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      /* save the remote side attributes, we will need it for the post SR */
      peer_[i].props = remote_con_data[i];
      peer_[i].connection.rnode.lid     = remote_con_data[i].lid;
      peer_[i].connection.rnode.alt_lid = remote_con_data[i].alt_lid;
      peer_[i].connection.remote.qpn    = remote_con_data[i].qp_num;
      peer_[i].connection.remote.psn    = remote_con_data[i].psn;
      peer_[i].connection.remote.rd_atomic = remote_con_data[i].rd_atomic;

      std::cout << "Peer " << i << std::endl;
      std::cout << remote_con_data[i];
#if 0
      if (config_.gid_idx >= 0) {
        std::cout << "  gid " << gid_t(peer_[i].props.gid);
      }
#endif
      std::cout << std::dec;
    }
  }

  // Sync before we move the QPs to the next state
  barrier();

  std::cerr << "Migrate QPs to RTR, RTS" << std::endl;
  for (::size_t i = 0; i < oob_.num_servers; ++i) {
    if (i == oob_.my_rank) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      if (rd_prep(&res_, &peer_[i].connection) != 0) {
        throw QPerfException("rd_prep peer " + to_string(i));
      }
    }
  }

  wc_.resize(post_send_chunk_);
  ::size_t q_size = std::max(cache_buffer_.capacity() / value_size,
                             write_buffer_.capacity() / value_size);
  ::size_t num_batches = (oob_.num_servers + batch_size_ - 1) / batch_size_;
  std::cerr << "Resize my queue pointers, use >= " << (num_batches * q_size * sizeof post_descriptor_[0][0] / 1048576.0) << "MB" << std::endl;
  post_descriptor_.resize(num_batches);
  for (::size_t i = 0; i < num_batches; ++i) {
    post_descriptor_[i].resize(q_size);
  }
  posts_.resize(num_batches);
}

template <typename T>
std::vector<const T*>& constify(std::vector<T*>& v) {
  // Compiler doesn't know how to automatically convert
  // std::vector<T*> to std::vector<T const*> because the way
  // the template system works means that in theory the two may
  // be specialised differently.  This is an explicit conversion.
  return reinterpret_cast<std::vector<const T*>&>(v);
}


int32_t DKVStoreRDMA::HostOf(DKVStoreRDMA::KeyType key) {
  if (include_master_) {
    return key % oob_.num_servers;
  } else {
    return 1 + key % (oob_.num_servers - 1);
  }
}

uint64_t DKVStoreRDMA::OffsetOf(DKVStoreRDMA::KeyType key) {
  if (include_master_) {
    return key / oob_.num_servers * value_size_;
  } else {
    return key / (oob_.num_servers - 1) * value_size_;
  }
}


::size_t DKVStoreRDMA::PollForCookies(::size_t current, ::size_t at_least, BatchTimer &timer) {
  while (current < at_least) {
    timer.finish.start();
    ::size_t n = rd_poll(&res_, wc_.data(), wc_.size());
    timer.finish.stop();
    for (::size_t i = 0; i < n; i++) {
      enum ibv_wc_status status = wc_[i].status;
      if (status != IBV_WC_SUCCESS) {
        throw QPerfException("rd_poll" + std::string(ibv_wc_status_str(status)));
      }
    }
    current += n;
  }

  return current;
}


void DKVStoreRDMA::post_batches(
    const std::vector<std::vector<PostDescriptor<DKVStoreRDMA::ValueType>>>
      &post_descriptor,
    const std::vector< ::size_t> &posts,
    uint32_t local_key,
    enum ibv_wr_opcode opcode,
    BatchTimer &timer) {
  ::size_t cookies = post_send_chunk_;
  ::size_t num_batches = (oob_.num_servers + batch_size_ - 1) / batch_size_;
  for (::size_t h = 0; h < num_batches; ++h) {
    ::size_t peer = (h + oob_.my_rank / num_batches) % num_batches;
    if (posts[peer] > 0) {      // correct statistics
      timer.host.start();
      for (::size_t i = 0; i < posts[peer]; ++i) {
        if (cookies == 0) {
          // Run out of cookies. Aqcuire at least one.
          cookies = PollForCookies(0, 1, timer);
        }

        timer.post.start();
        const auto *d = &post_descriptor[peer][i];
        if (rd_post_rdma_std(d->connection_,
                             local_key, // cache_.region_.mr->lkey,
                             d->rkey_,
                             opcode,
                             1,
                             reinterpret_cast<void* const*>(&d->local_addr_),
                             reinterpret_cast<const void* const*>(&d->remote_addr_),
                             &d->sizes_) != 0) {
          throw QPerfException("rd_post_rdma_std");
        }
        timer.post.stop();
        num_posts_++;
        cookies--;
      }
      timer.host.stop();
    }
    if (false) {
      std::cerr << "For now, sync to see if it helps throughput..." << std::endl;
      barrier();
    }
  }

  // Collect the remaining cookies
  PollForCookies(cookies, post_send_chunk_, timer);
}


void DKVStoreRDMA::ReadKVRecords(std::vector<DKVStoreRDMA::ValueType *> &cache,
                                 const std::vector<KeyType> &key,
                                 RW_MODE::RWMode rw_mode) {
  if (rw_mode != RW_MODE::READ_ONLY) {
    std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
  }

  t_read_.outer.start();

  for (auto &s : posts_) {
    s = 0;
  }
  for (::size_t i = 0; i < key.size(); i++) {
    ::size_t owner = HostOf(key[i]);
    if (owner == oob_.my_rank) {
      // FIXME: do this asynchronously
      t_read_.local.start();
      // Read directly, without RDMA
      cache[i] = value_.area_ + OffsetOf(key[i]);

      bytes_local_read += value_size_ * sizeof(ValueType);
      t_read_.local.stop();

    } else {
      ValueType *target = cache_buffer_.get(value_size_);
      cache[i] = target;

      ::size_t batch = owner / batch_size_;
      ::size_t n = posts_[batch];
      posts_[batch] += 1;
      assert(post_descriptor_.capacity() > batch);
      assert(post_descriptor_[batch].capacity() > n);

      auto *d = &post_descriptor_[batch][n];
      d->connection_ = &peer_[owner].connection;
      d->rkey_ = peer_[owner].props.value_rkey;
      d->local_addr_ = target;
      d->sizes_ = value_size_ * sizeof(ValueType);
      d->remote_addr_ = (const ValueType *)(peer_[owner].props.value +
                                            OffsetOf(key[i]) * sizeof(ValueType));

      bytes_remote_read += value_size_ * sizeof(ValueType);
    }
  }

  post_batches(post_descriptor_, posts_, cache_.region_.mr->lkey,
               IBV_WR_RDMA_READ, t_read_);

  t_read_.outer.stop();
}


void DKVStoreRDMA::WriteKVRecords(const std::vector<KeyType> &key,
                                  const std::vector<const ValueType *> &value) {
  t_write_.outer.start();

  for (auto &s : posts_) {
    s = 0;
  }
  for (::size_t i = 0; i < key.size(); i++) {
    ::size_t owner = HostOf(key[i]);
    if (owner == oob_.my_rank) {
      // FIXME: do this asynchronously
      t_write_.local.start();
      // Write directly, without RDMA
      ValueType *target = value_.area_ + OffsetOf(key[i]);
      memcpy(target, value[i], value_size_ * sizeof(ValueType));
      t_write_.local.stop();

      bytes_local_written += value_size_ * sizeof(ValueType);

    } else {
      const ValueType *source;
      if (write_.contains(value[i])) {
        source = value[i];
      } else {
        ValueType *v = write_buffer_.get(value_size_);
        memcpy(v, value[i], value_size_ * sizeof(ValueType));
        source = v;
      }
      assert(source >= write_buffer_.buffer());
      assert(source + value_size_ <= write_buffer_.buffer() + write_buffer_.capacity());

      ::size_t batch = owner / batch_size_;
      ::size_t n = posts_[batch];
      posts_[batch] += 1;
      assert(post_descriptor_.capacity() > batch); 
      assert(post_descriptor_[batch].capacity() > n);

      auto *d = &post_descriptor_[batch][n];
      d->connection_ = &peer_[owner].connection;
      d->rkey_ = peer_[owner].props.value_rkey;
      d->local_addr_ = const_cast<ValueType *>(source);       // sorry, API
      d->sizes_ = value_size_ * sizeof(ValueType);
      d->remote_addr_ = (const ValueType *)(peer_[owner].props.value +
                                            OffsetOf(key[i]) * sizeof(ValueType));

      bytes_remote_written += value_size_ * sizeof(ValueType);
    }
  }

  post_batches(post_descriptor_, posts_, write_.region_.mr->lkey,
               IBV_WR_RDMA_WRITE, t_write_);

  t_write_.outer.stop();
}


std::vector<DKVStoreRDMA::ValueType *>
    DKVStoreRDMA::GetWriteKVRecords(::size_t n) {
  std::vector<ValueType *> w(n);
  for (::size_t i = 0; i < n; i++) {
    w[i] = write_buffer_.get(value_size_);
  }

  return w;
}


void DKVStoreRDMA::FlushKVRecords(const std::vector<KeyType> &key) {
  std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
  write_buffer_.reset();
}

/**
 * Purge the cache area
 */
void DKVStoreRDMA::PurgeKVRecords() {
  cache_buffer_.reset();
  write_buffer_.reset();
}

void DKVStoreRDMA::barrier() {
  oob_network_.barrier();
}

}   // namespace DKV
}   // namespace DKVRDMA