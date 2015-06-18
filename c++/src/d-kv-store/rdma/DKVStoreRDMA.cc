/*
 * Copyright notice
 */
#ifdef ENABLE_RDMA
#include <d-kv-store/rdma/DKVStoreRDMA.h>
#endif

#include <unistd.h>

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

#ifdef ENABLE_RDMA
#include <infiniband/verbs.h>
#endif


namespace DKV {
namespace DKVRDMA {

struct ibv_device **global_dev_list = NULL;

#ifdef USE_MPI

void DKVStoreRDMA::mpi_error_test(int r, const std::string &message) {
  if (r != MPI_SUCCESS) {
	std::cerr << "It throws me error code " << r << std::endl;
    throw NetworkException("MPI error " + r + message);
  }
}

void DKVStoreRDMA::init_networking() {
  int r;
  if (! mpi_initialized) {
    std::cerr << "FIXME MPI_Init reentrant" << std::endl;
    r = MPI_Init(NULL, NULL);
    mpi_error_test(r, "MPI_Init() fails");
  }
  int n;
  r = MPI_Comm_size(MPI_COMM_WORLD, &n);
  mpi_error_test(r, "MPI_Comm_size() fails");
  num_servers_ = n;
  r = MPI_Comm_rank(MPI_COMM_WORLD, &n);
  mpi_error_test(r, "MPI_Comm_rank() fails");
  my_rank_ = n;
}

void DKVStoreRDMA::alltoall(const void *sendbuf, ::size_t send_item_size,
                            void *recvbuf, ::size_t recv_item_size) {
  int r;
  r = MPI_Alltoall(const_cast<void *>(sendbuf), send_item_size, MPI_BYTE,
                   recvbuf, recv_item_size, MPI_BYTE,
                   MPI_COMM_WORLD);
  mpi_error_test(r, "MPI conn alltoall");
}


void DKVStoreRDMA::barrier() {
  int r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "barrier after qp exchange");
}

#else

void DKVStoreRDMA::init_networking() {
  std::cerr << "hardwire network = " << "socket" << ", interface = " << oob_interface_ << std::endl;

  mr::OptionList options;
  options.push_back(mr::Option("interface", oob_interface_));
  // network_ = mr::net::NetworkImpl::createNetwork("socket", options);
  network_ = new mr::net::SocketNetwork(options);
  network_->createAllToAllConnections(rw_list_, network_->getPeers(),
                                      mr::net::network_type::INTERMEDIATE, true);
  num_servers_ = rw_list_.size();             
  int i = 0;
  for (auto & iter : rw_list_) {
    if (iter.writer == NULL && iter.reader == NULL) {
      my_rank_ = i;
    }
    i++;    
  }

  broadcast_.Init(network_);
}


void DKVStoreRDMA::alltoall_leaf(const char *sendbuf, ::size_t send_item_size,
                                 char *recvbuf, ::size_t recv_item_size,
                                 ::size_t me, ::size_t size,
                                 ::size_t start, ::size_t size_2pow) {
  const bool FAKE = false;

  if (me < start + size_2pow) {
    for (::size_t p = 0; p < size_2pow; p++) {
      ::size_t peer = start + size_2pow + (me + p) % size_2pow;
      if (FAKE) {
        std::cerr << me << ": subsize " << size_2pow << " start " << start << " " << ((peer >= size) ? "dont" : "") << " read/write to " << peer << std::endl;
      } else if (peer < size) {
        rw_list_[peer].reader->readFully(recvbuf + peer * recv_item_size,
                                         recv_item_size);
        rw_list_[peer].writer->write(sendbuf + peer * send_item_size,
                                     send_item_size);
      }
    }
  } else {
    for (::size_t p = 0; p < size_2pow; p++) {
      ::size_t peer = start + (me - p) % size_2pow;
      if (FAKE) {
        std::cerr << me << ": subsize " << size_2pow << " start " << start << " " << std::string((peer >= size) ? "dont" : "") << " write/read to " << peer << std::endl;
      } else if (peer < size) {
        rw_list_[peer].writer->write(sendbuf + peer * send_item_size,
                                     send_item_size);
        rw_list_[peer].reader->readFully(recvbuf + peer * recv_item_size,
                                         recv_item_size);
      }
    }
  }
}


void DKVStoreRDMA::alltoall_DC(const char *sendbuf, ::size_t send_item_size,
                               char *recvbuf, ::size_t recv_item_size,
                               ::size_t me, ::size_t size,
                               ::size_t start, ::size_t size_2pow) {
  if (size_2pow < 2) {
    return;
  }

  size_2pow /= 2;

  alltoall_leaf(sendbuf, send_item_size,
                recvbuf, recv_item_size,
                me, size, start, size_2pow);

  // Divide and conquer
  if (me < start + size_2pow) {
    alltoall_DC(sendbuf, send_item_size,
                recvbuf, recv_item_size,
                me, size, start, size_2pow);
  } else {
    alltoall_DC(sendbuf, send_item_size,
                recvbuf, recv_item_size,
                me, size, start + size_2pow, size_2pow);
  }
}


void DKVStoreRDMA::alltoall(const void *sendbuf, ::size_t send_item_size,
                            void *recvbuf, ::size_t recv_item_size) {
  ::size_t size_2pow = mr::next_2_power(num_servers_);

  alltoall_DC(static_cast<const char *>(sendbuf), send_item_size,
              static_cast<char *>(recvbuf), recv_item_size,
              my_rank_, num_servers_, 0, size_2pow);

  // copy our own contribution by hand
  memcpy(static_cast<char *>(recvbuf) + my_rank_ * recv_item_size,
         static_cast<const char *>(sendbuf) + my_rank_ * send_item_size,
         recv_item_size);
}


void DKVStoreRDMA::barrier() {
  t_barrier_.start();
  int32_t dummy = -1;
  if (my_rank_ == 0) {
    broadcast_.bcast_send(dummy);
    dummy = broadcast_.reduce_rcve<int32_t>();
  } else {
    broadcast_.bcast_rcve(&dummy);
    broadcast_.reduce_send(dummy);
  }
  t_barrier_.stop();
}

#endif

/*
 * Class description
 */
DKVStoreRDMA::DKVStoreRDMA() {
}

DKVStoreRDMA::~DKVStoreRDMA() {
#ifdef USE_MPI
  if (! mpi_initialized) {
    MPI_Finalize();
  }
#else
  broadcast_.Finish();

  for (auto & c : rw_list_) {
    if (c.writer != NULL) {
      c.writer->close();
    }
  }
  for (auto & c : rw_list_) {
    delete c.reader;
    delete c.writer;
  }
  delete network_;
#endif

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

  po::options_description desc("RDMA options");
  desc.add_options()
    // ("rdma.oob-port", po::value(&config_.tcp_port)->default_value(0), "RDMA out-of-band TCP port")
    ("rdma:dev", 
     po::value(&dev_name_),
     "RDMA device name")
    ("rdma:port",
     po::value(&ib_port),
     "RDMA device port")
    ("rdma:mtu",
     po::value(&mtu)->default_value(2048),
     "RDMA MTU (256/512/1024/2048/4096) (512)")
    ("rdma:chunk",
     po::value(&post_send_chunk_),
     "RDMA max number of messages per post")
    ("rdma:peers",
     po::value(&batch_size_)->default_value(0),
     "RDMA max number of peers to address in one post")
    // ("rdma:oob-network",
    // po::value(&oob_impl_)->default_value("socket"),
    // "RDMA OOB network implementation")
#ifdef USE_MPI
    ("rdma:mpi-initialized",
     po::bool_switch()->default_value(false),
     "MPI is already initialized")
#else
    ("rdma:oob-interface",
     po::value(&oob_interface_)->default_value("ib0"),
     "RDMA OOB network interface")
#endif
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

#ifdef USE_MPI
  if (vm.count("rdma:mpi-initialized") > 0) {
    mpi_initialized = true;
  }
#endif

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

  init_networking();  // along the way, defines my_rank_ and num_servers_

  if (rd_open(&res_, IBV_QPT_RC, post_send_chunk_, 0) != 0) {
    throw QPerfException("rd_open");
  }

  if (batch_size_ == 0) {
    if (num_servers_ <= 3) {
      batch_size_ = num_servers_;
    } else if (num_servers_ <= 6) {
      batch_size_ = 4;
    } else if (num_servers_ <= 12) {
      batch_size_ = 5;
    } else if (num_servers_ <= 24) {
      batch_size_ = 6;
    } else if (num_servers_ <= 48) {
      batch_size_ = 7;
    } else {
      batch_size_ = 8;
    }
  }

  value_size_ = value_size;
  total_values_ = total_values;
  keys_per_host_ = (total_values_ + num_servers_ - 1) / num_servers_;

  /* memory buffer to hold the value data */
  ::size_t my_values = (total_values + num_servers_ - 1) / num_servers_;
  value_.Init(&res_, my_values * value_size);
  std::cout << "MR/value " << value_ << std::endl;

  /* memory buffer to hold the cache data */
  cache_.Init(&res_, &cache_buffer_, max_cache_capacity * value_size);
  std::cout << "MR/cache " << cache_ << std::endl;

  /* memory buffer to hold the zerocopy write data */
  write_.Init(&res_, &write_buffer_, max_write_capacity * value_size);
  std::cout << "MR/write " << write_ << std::endl;

  peer_.resize(num_servers_);
  for (::size_t i = 0; i < num_servers_; i++) {
    if (i == my_rank_) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      peer_[i].Init(&res_);
    }
  }

  /* exchange using o-o-b network info required to connect QPs */
  std::vector<cm_con_data_t> my_con_data(num_servers_);
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
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

  std::vector<cm_con_data_t> remote_con_data(num_servers_);
  alltoall(my_con_data.data(), sizeof my_con_data[0],
           remote_con_data.data(), sizeof remote_con_data[0]);

  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
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
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      // std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      if (rd_prep(&res_, &peer_[i].connection) != 0) {
        throw QPerfException("rd_prep peer " + to_string(i));
      }
    }
  }

  if (batch_size_ > num_servers_) {
    batch_size_ = num_servers_;
  }

  wc_.resize(post_send_chunk_);
  ::size_t q_size = std::max(cache_buffer_.capacity() / value_size,
                             write_buffer_.capacity() / value_size);
  ::size_t num_batches = (num_servers_ + batch_size_ - 1) / batch_size_;
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


int32_t DKVStoreRDMA::HostOf(DKVStoreInterface::KeyType key) {
  // return key % num_servers_;
  // Nowadays, prefer blocked over striped
  return key / keys_per_host_;
}

uint64_t DKVStoreRDMA::OffsetOf(DKVStoreInterface::KeyType key) {
  // return key / num_servers_ * value_size_;
  // Nowadays, prefer blocked over striped
  return (key - HostOf(key) * keys_per_host_) * value_size_;
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
  ::size_t num_batches = (num_servers_ + batch_size_ - 1) / batch_size_;
  for (::size_t h = 0; h < num_batches; ++h) {
    ::size_t peer = (h + my_rank_ / num_batches) % num_batches;
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
    if (owner == my_rank_) {
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
    if (owner == my_rank_) {
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

}   // namespace DKV
}   // namespace DKVRDMA
