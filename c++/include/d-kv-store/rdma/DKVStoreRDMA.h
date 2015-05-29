/*
 * Copyright
 */
#ifndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
#define APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__

// #define OK_TO_DEFINE_IN_CC_FILE
#ifdef OK_TO_DEFINE_IN_CC_FILE
#define VIRTUAL virtual
#else
#define VIRTUAL
#endif

#include <errno.h>
#include <string.h>
#include <inttypes.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <infiniband/verbs.h>

#include <mcmc/timer.h>
#include <mr/timer.h>

// #define USE_MPI

#ifdef USE_MPI
#  include <mpi.h>
#else
#  include <mr/net/netio.h>
#  include <mr/net/broadcast.h>
#  include <../src/mr/net/sockets/netio_sockets.h>
#endif

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

#include <d-kv-store/DKVStore.h>

#include <d-kv-store/rdma/qperf-rdma.h>


#if 0
inline void check_dev_list(const char *file, int line) {
    extern ::ibv_device **global_dev_list;

    int num;
    std::cerr << file << "." << line <<
         ": dev_list[0] " << (void *)global_dev_list[0] <<
         " ibv_get_device_list()[0] " << ::ibv_get_device_list(&num)[0] <<
         " " << global_dev_list[0]->dev_name << std::endl;
}
#endif


namespace DKV {
namespace DKVRDMA {

using ::mr::timer::Timer;


template <typename T>
std::string to_string(T value) {
  std::ostringstream s;
  s << value;
  return s.str();
}


class RDMAException : public DKVException {
 public:
  RDMAException(int rc, const std::string &reason) throw() :
      DKVException((rc == 0) ? reason
                             : (reason + ": " + std::string(strerror(rc)))) {
  }

  RDMAException(const std::string &reason) throw() : DKVException(reason) {
  }

  virtual ~RDMAException() throw() {
  }

 protected:
  // RDMAException() throw() : reason_("<apparently inherited>") {
  // }
};

class QPerfException : public RDMAException {
 public:
#if 0
  QPerfException() throw() :
      RDMAException(errno, rd_get_error_message()) {
  }
#endif

  QPerfException(const std::string &reason) throw()
      : RDMAException(errno, reason + " " + std::string(rd_get_error_message())) {
    std::cerr << "QPerfException(" << reason << ") errno " << strerror(errno) << " QPerf " << rd_get_error_message() << std::endl;
  }

  virtual ~QPerfException() throw() {
  }
};


class NetworkException : public RDMAException {
 public:
  NetworkException(const std::string &reason) throw() : RDMAException(reason) {
  }

  virtual ~NetworkException() throw() {
  }
};


class BatchTimer {
 public:
  Timer outer;
  Timer local;
  Timer host;
  Timer post;
  Timer finish;
};


struct cm_con_data_t {
  uint64_t		value;
  uint64_t		cache;
  uint32_t		value_rkey;
  uint32_t		cache_rkey;
  uint32_t		qp_num;
  uint32_t		psn;
  uint32_t              rd_atomic;
  uint16_t		lid;
  uint16_t		alt_lid;

  std::ostream &put(std::ostream &s) const {
    std::ios_base::fmtflags flags = s.flags();
    s << std::hex;
    s << "  value address = 0x" << value << std::endl;
    s << "  value rkey = 0x" << value_rkey << std::endl;
    s << "  cache address = 0x" << cache << std::endl;
    s << "  cache rkey = 0x" << cache_rkey << std::endl;
    s << std::dec;
    s << "  QP number = " << qp_num << std::endl;
    s << "  PSN number = " << psn << std::endl;
    s << "  max.atomic = " << rd_atomic << std::endl;
    s << "  LID = 0x" << lid << std::endl;
    s.flags(flags);

    return s;
  }
};

inline std::ostream &operator<< (std::ostream &s, const cm_con_data_t &r) {
  return r.put(s);
}


struct rdma_peer {
  CONNECTION     connection;          /* QP handle */
  struct cm_con_data_t props;         /* remote side properties */

  ~rdma_peer() {
    if (rd_close_qp(&connection) != 0) {
      throw QPerfException("rd_close_qp");
    }
  }

  void Init(DEVICE *device) {
    std::cerr << "Create QP" << std::endl;
    if (rd_create_qp(device, &connection, device->ib.context, NULL) != 0) {
      throw QPerfException("rd_create_qp");
    }
    std::cerr << "Migrate this QP to INIT" << std::endl;
    if (rd_open_2(device, &connection) != 0) {
      throw QPerfException("rd_open_2");
    }
  }
};


template <typename ValueType>
class rdma_area {
 public:
  rdma_area() : n_elements_(0), area_(NULL) {
    memset(&region_, 0, sizeof region_);
  }

  ~rdma_area() {
    if (rd_mrfree(&region_, res_) != 0) {
      throw QPerfException("rd_mrfree");
    }
  }

  void Init(const DEVICE *device, ::size_t n_elements) {
    res_ = device;
    n_elements_ = n_elements;

    /* allocate the memory buffer that will hold the data */
    if (rd_mralloc(&region_, device, n_elements * sizeof(ValueType)) != 0) {
      throw QPerfException("rd_mralloc");
    }
    area_ = reinterpret_cast<ValueType *>(region_.vaddr);
  }

  void Init(const DEVICE *device, Buffer<ValueType> *buffer, ::size_t n_elements) {
    Init(device, n_elements);
    buffer->Init(area_, n_elements * sizeof(ValueType));
  }

  bool contains(const ValueType *v) const {
    return (v >= area_ && v < area_ + n_elements_);
  }

  std::ostream &put(std::ostream &s) const {
    std::ios_base::fmtflags flags = s.flags();
    s << std::hex;
    s << "addr=" << (void *)area_;
    s << " size " << (n_elements_ * sizeof(ValueType));
    s << std::hex;
    s << ", lkey=0x" << region_.mr->lkey;
    s << ", rkey=0x" << region_.mr->rkey;
    s << ", flags=0x" << (IBV_ACCESS_LOCAL_WRITE  |
                          IBV_ACCESS_REMOTE_READ  |
                          IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_ATOMIC);
    s << ", mr=" << (void *)&region_.mr;
    s.flags(flags);

    return s;
  }

 private:
  const DEVICE *res_;

 public:
  ::size_t n_elements_;         // alias for ease of use
  ValueType *area_;             // alias for ease of use
  REGION region_;
};

template <typename ValueType>
inline std::ostream &operator<< (std::ostream &s,
                                 const rdma_area<ValueType> &r) {
  return r.put(s);
}


template <typename ValueType>
class PostDescriptor {
 public:
  CONNECTION *connection_;
  uint32_t rkey_;
  ValueType *local_addr_;
  ::size_t sizes_;
  const ValueType *remote_addr_;
};

/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  DKVStoreRDMA() : DKVStoreInterface() {
  }

  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

#ifndef OK_TO_DEFINE_IN_CC_FILE
  virtual ~DKVStoreRDMA() {
#ifdef USE_MPI
    MPI_Finalize();
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
#else
  virtual ~DKVStoreRDMA();
#endif

 private:
  static double GBs_from_time(
      const std::chrono::high_resolution_clock::duration &dt, int64_t bytes) {
    double gbs;
    double s = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count() / 1000000000.0;

    gbs = bytes / s / (1LL << 30);

    return gbs;
  }

  static double GBs_from_time(const double &dt, int64_t bytes) {
    double gbs;
    gbs = bytes / dt / (1LL << 30);

    return gbs;
  }

  static double GBs_from_timer(const Timer &timer, int64_t bytes) {
    return GBs_from_time(timer.total(), bytes);
  }

 public:
#ifndef OK_TO_DEFINE_IN_CC_FILE
  VIRTUAL void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
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
#ifndef USE_MPI
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


  VIRTUAL void ReadKVRecords(std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode) {
    if (rw_mode != RW_MODE::READ_ONLY) {
      std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
    }

    t_read_.outer.start();

#ifndef DISABLE_INFINIBAND
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
#endif

    t_read_.outer.stop();
  }


  VIRTUAL void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) {
    t_write_.outer.start();

#ifndef DISABLE_INFINIBAND
    // Fill the linked lists of SGE requests
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
#endif

    t_write_.outer.stop();
  }


  VIRTUAL std::vector<ValueType *> GetWriteKVRecords(::size_t n) {
    std::vector<ValueType *> w(n);
    for (::size_t i = 0; i < n; i++) {
      w[i] = write_buffer_.get(value_size_);
    }

    return w;
  }


  VIRTUAL void FlushKVRecords(const std::vector<KeyType> &key) {
    std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
    write_buffer_.reset();
  }

  /**
   * Purge the cache area
   */
  VIRTUAL void PurgeKVRecords() {
    cache_buffer_.reset();
    write_buffer_.reset();
  }

 private:
  int32_t HostOf(DKVStoreRDMA::KeyType key) {
    return key % num_servers_;
  }

  uint64_t OffsetOf(DKVStoreRDMA::KeyType key) {
    return key / num_servers_ * value_size_;
  }

  ::size_t PollForCookies(::size_t current, ::size_t at_least, BatchTimer &timer) {
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

  void post_batches(const std::vector<std::vector<PostDescriptor<ValueType>>> &post_descriptor,
                    const std::vector< ::size_t> &posts,
                    uint32_t local_key,
                    enum ibv_wr_opcode opcode,
                    BatchTimer &timer) {
    ::size_t cookies = post_send_chunk_;
    ::size_t num_batches = (num_servers_ + batch_size_ - 1) / batch_size_;
    for (::size_t h = 0; h < num_batches; ++h) {
      ::size_t peer = (h + my_rank_ / num_batches) % num_batches;
      timer.host.start();
      for (::size_t i = 0; i < posts[peer]; ++i) {
        if (cookies == 0) {
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
      if (false) {
        std::cerr << "For now, sync to see if it helps throughput..." << std::endl;
        barrier();
      }
    }

    PollForCookies(cookies, post_send_chunk_, timer);
  }
#endif

#ifdef OK_TO_DEFINE_IN_CC_FILE
  VIRTUAL void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args);

  VIRTUAL void ReadKVRecords(std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode);

  VIRTUAL void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  VIRTUAL void FlushKVRecords(const std::vector<KeyType> &key);

  std::vector<ValueType *> GetWriteKVRecords(::size_t n);

  /**
   * Purge the cache area
   */
  VIRTUAL void PurgeKVRecords();

 private:
  int32_t HostOf(DKVStoreRDMA::KeyType key);
  uint64_t OffsetOf(DKVStoreRDMA::KeyType key);

  void modify_qp_to_init(struct ::ibv_qp *qp);
  void modify_qp_to_rtr(struct ::ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
                        const gid_t &dgid);
  void modify_qp_to_rts(struct ::ibv_qp *qp);
  void post_receive(struct rdma_area<ValueType> *area_, struct ::ibv_qp *qp);
  void connect_qp();
  void check_ibv_status(int r, const struct ::ibv_send_wr *bad);
  void finish_completion_queue(::size_t n, ::ibv_wc_opcode opcode);

  void init_networking();
#ifndef USE_MPI
  void alltoall_leaf(const char *sendbuf, ::size_t send_item_size,
                     char *recvbuf, ::size_t recv_item_size,
                     ::size_t me, ::size_t size,
                     ::size_t start, ::size_t size_2pow);
  void alltoall_DC(const char *sendbuf, ::size_t send_item_size,
                   char *recvbuf, ::size_t recv_item_size,
                   ::size_t me, ::size_t size,
                   ::size_t start, ::size_t size_2pow);
#endif
 public:
  void alltoall(const void *sendbuf, ::size_t send_item_size,
                void *recvbuf, ::size_t recv_item_size);
  void barrier();
#endif


#ifndef OK_TO_DEFINE_IN_CC_FILE
  void check_ibv_status(int r, const struct ::ibv_send_wr *bad) {
    if (r != 0) {
      std::cerr << "IBV status " << r << " bad " << bad << std::endl;
    }
  }


  void finish_completion_queue(::size_t n, ::ibv_wc_opcode opcode) {
#ifndef DISABLE_INFINIBAND
    if (wc_.capacity() < n) {
      wc_.reserve(n);
    }

    int r;
    ::size_t seen = 0;
    while (seen < n) {
      t_poll_cq_.start();
      do {

        r = ::ibv_poll_cq(res_.cq, n, wc_.data() + seen);
        if (r < 0) {
          std::cerr << "Oops, must handle wrong result cq poll" << std::endl;
          break;
        }
      } while (r == 0);
      t_poll_cq_.stop();
      for (::size_t i = seen; i < seen + r; i++) {
        if (wc_[i].status != IBV_WC_SUCCESS) {
          throw RDMAException("ibv_poll_cq status " + to_string(wc_[r].status) +
                              " " +
                              std::string(ibv_wc_status_str(wc_[r].status)));
        }
        assert(wc_[i].opcode == opcode);
      }
      seen += r;
    }
#endif
  }

#ifdef USE_MPI

  static void mpi_error_test(int r, const std::string &message) {
    if (r != MPI_SUCCESS) {
      throw NetworkException("MPI error " + r + message);
    }
  }

  void init_networking() {
    std::cerr << "FIXME MPI_Init reentrant" << std::endl;
    int r;
    r = MPI_Init(NULL, NULL);
    mpi_error_test(r, "MPI_Init() fails");
    int n;
    r = MPI_Comm_size(MPI_COMM_WORLD, &n);
    mpi_error_test(r, "MPI_Comm_size() fails");
    num_servers_ = n;
    r = MPI_Comm_rank(MPI_COMM_WORLD, &n);
    mpi_error_test(r, "MPI_Comm_rank() fails");
    my_rank_ = n;
  }

 public:
  void alltoall(const void *sendbuf, ::size_t send_item_size,
                              void *recvbuf, ::size_t recv_item_size) {
    int r;
    r = MPI_Alltoall(const_cast<void *>(sendbuf), send_item_size, MPI_BYTE,
                     recvbuf, recv_item_size, MPI_BYTE,
                     MPI_COMM_WORLD);
    mpi_error_test(r, "MPI conn alltoall");
  }


  void barrier() {
    int r = MPI_Barrier(MPI_COMM_WORLD);
    mpi_error_test(r, "barrier after qp exchange");
  }
private:

#else   // def USE_MPI

  void init_networking() {
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


 private:
  void alltoall_leaf(const char *sendbuf, ::size_t send_item_size,
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


  void alltoall_DC(const char *sendbuf, ::size_t send_item_size,
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


 public:
  void alltoall(const void *sendbuf, ::size_t send_item_size,
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


  void barrier() {
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

#endif  // def USE_MPI
#endif  // ndef OK_TO_USE_CC_FILE

 private:
  ::size_t num_servers_;
  ::size_t my_rank_;
  
  DEVICE res_;
  std::vector<rdma_peer> peer_;

  std::vector< ::ibv_wc> wc_;

  std::vector<std::vector<PostDescriptor<ValueType>>> post_descriptor_;
  std::vector< ::size_t> posts_;
  ::size_t batch_size_;

  std::string dev_name_;
  ::size_t post_send_chunk_ = 1024;

  /* memory buffer pointers, used for RDMA and send ops */
  rdma_area<ValueType> value_;
  rdma_area<ValueType> cache_;
  rdma_area<ValueType> write_;

#ifndef USE_MPI
  std::string oob_impl_;
  std::string oob_interface_;
  mr::net::Network *network_;
  mr::net::Broadcast broadcast_;
  mr::net::Network::RWList rw_list_;
#endif

  Timer t_poll_cq_;
  BatchTimer t_read_;
  BatchTimer t_write_;
  Timer t_barrier_;
  ::size_t msgs_per_post_ = 0;
  ::size_t num_posts_ = 0;

  int64_t bytes_local_read = 0;
  int64_t bytes_remote_read = 0;
  int64_t bytes_local_written = 0;
  int64_t bytes_remote_written = 0;
};

}   // namespace DKVRDMA
}   // namespace DKV

#endif  // ndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
