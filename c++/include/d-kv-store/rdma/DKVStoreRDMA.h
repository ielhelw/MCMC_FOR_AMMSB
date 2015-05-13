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

#ifndef USE_MPI
#ifndef DISABLE_NETWORKING
#include <mr/net/netio.h>
#include <mr/net/broadcast.h>
#include <../src/mr/net/sockets/netio_sockets.h>
#endif
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

extern struct ibv_device **global_dev_list;

#ifdef DEBUG_CC_FILE_H_FILE_FIASCO
#define CHECK_DEV_LIST() \
  do { \
    int num; \
    std::cerr << __FILE__ << "." << __LINE__ << \
         ": dev_list[0] " << (void *)::DKV::DKVRDMA::global_dev_list[0] << \
         " ibv_get_device_list()[0] " << ::ibv_get_device_list(&num)[0] << \
         " " << ::DKV::DKVRDMA::global_dev_list[0]->dev_name << std::endl; \
    /* check_dev_list(__FILE__, __LINE__); */ \
  } while (0)
#else
#define CHECK_DEV_LIST() \
  do { } while (0)
#endif


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
  QPerfException() throw() :
      RDMAException(errno, rd_get_error_message()) {
  }

  QPerfException(const std::string &reason) throw()
      : RDMAException(errno, reason + " " + rd_get_error_message()) {
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


struct q_item_t {
  q_item_t() {
    wr.sg_list = &sg_list;
  }

  struct ::ibv_send_wr    wr;
  struct ::ibv_sge        sg_list;
  q_item_t             *next;
};


struct cm_con_data_t {
  uint64_t		value;
  uint64_t		cache;
  uint32_t		value_rkey;
  uint32_t		cache_rkey;
  uint32_t		qp_num;
  uint32_t		psn;
  uint16_t		lid;
  uint16_t		alt_lid;

  std::ostream &put(std::ostream &s) const {
    std::ios_base::fmtflags flags = s.flags();
    s << std::hex;
    s << "  value address = 0x" << value << std::endl;
    s << "  value rkey = 0x" << value_rkey << std::endl;
    s << "  cache address = 0x" << cache << std::endl;
    s << "  cache rkey = 0x" << cache_rkey << std::endl;
    s << "  QP number = 0x" << qp_num << std::endl;
    s << "  PSN number = 0x" << psn << std::endl;
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


/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  DKVStoreRDMA() : DKVStoreInterface(), REQUIRE_POSTED_RECEIVE(false) {
CHECK_DEV_LIST();
  }

  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

#ifndef OK_TO_DEFINE_IN_CC_FILE
  virtual ~DKVStoreRDMA() {
#ifndef DISABLE_NETWORKING
#  ifndef USE_MPI
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
#  endif
#endif
    Timer::printHeader(std::cout);
    std::cout << t_poll_cq << std::endl;
    std::cout << t_read << std::endl;
    std::cout << t_local_read << std::endl;
    std::cout << t_post_read << std::endl;
    std::cout << t_finish_read << std::endl;
    std::cout << t_write << std::endl;
    std::cout << t_local_write << std::endl;
    std::cout << t_post_write << std::endl;
    std::cout << t_finish_write << std::endl;
    std::cout << t_barrier << std::endl;
    // std::chrono::high_resolution_clock::duration dt;
    std::cout << "Local read   " << bytes_local_read     << "B " <<
      GBs_from_timer(t_local_read, bytes_local_read) << "GB/s" << std::endl;
    auto dt = t_read.total() - t_local_read.total();
    std::cout << "Remote read  " << bytes_remote_read    << "B " <<
      GBs_from_time(dt, bytes_remote_read) << "GB/s" << std::endl;
    std::cout << "Local write  " << bytes_local_written  << "B " <<
      GBs_from_timer(t_local_write, bytes_local_written) << "GB/s" << std::endl;
    dt = t_write.total() - t_local_write.total();
    std::cout << "Remote write " << bytes_remote_written << "B " <<
      GBs_from_time(dt, bytes_remote_written) << "GB/s" << std::endl;

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

  virtual void InfoH() const {
    CHECK_DEV_LIST();
  }

  virtual void Info() const;

 private:
  void post_linked_list(::ibv_qp *qp, ::ibv_send_wr *q_front,
                        ::size_t peer, ::ibv_wc_opcode opcode) {
    // Only for verbose
    std::string action;
    switch (opcode) {
    case IBV_WC_RDMA_READ:
      action = "read";
      break;
    case IBV_WC_RDMA_WRITE:
      action = "write";
      break;
    default:
      action = "<any old post>";
      break;
    }

    if (peer == my_rank_) {
      // already done the memcpy
      // std::cerr << "Skip the home " << action << "s" << std::endl;
      return;
    }

    if (true) {
      if (false) {
        ::size_t sent_items = 0;
        for (auto p = q_front; p != NULL; p = p->next) {
          sent_items++;
        }

        std::cerr << "post " << sent_items << " " << action << " requests to host " << peer << std::endl;
      }

      // std::cerr << "******** Do the " << action << "s in chunks of size " << post_send_chunk_ << std::endl;
      ::ibv_send_wr *front = q_front;
      while (front != NULL) {
        ::ibv_send_wr *next_front;
        ::ibv_send_wr *p = front;
        ::size_t count = 0;
        while (true) {
          if (p == NULL) {
            next_front = NULL;
            break;
          }
          count++;
          if (count == post_send_chunk_) {
            next_front = p->next;
            p->send_flags |= IBV_SEND_SIGNALED;
            p->next = NULL;
            break;
          }
          if (p->next == NULL) {
            p->send_flags |= IBV_SEND_SIGNALED;
          } else {
            assert(! (p->send_flags & IBV_SEND_SIGNALED));
          }
          p = p->next;
        }

        // std::cerr << "post " << count << " subchunk requests to host " << peer << std::endl;

        if (front != NULL) {
          struct ::ibv_send_wr *bad;

          if (opcode == IBV_WC_RDMA_READ) {
            t_post_read.start();
          } else {
            t_post_write.start();
          }
          int r = ::ibv_post_send(qp, front, &bad);
          check_ibv_status(r, bad);
          if (opcode == IBV_WC_RDMA_READ) {
            t_post_read.stop();
            t_finish_read.start();
          } else {
            t_post_write.stop();
            t_finish_write.start();
          }

          // finish_completion_queue(count, opcode);
          finish_completion_queue(1, opcode);
          if (opcode == IBV_WC_RDMA_READ) {
            t_finish_read.stop();
          } else {
            t_finish_write.stop();
          }
        }

        front = next_front;
      }

    } else if (false) {
      ::size_t sent_items = 0;
      for (auto p = q_front; p != NULL; p = p->next) {
        sent_items++;
      }

      if (sent_items == 0) {
        std::cerr << "skip posting " << sent_items << " " << action << " requests to host " << peer << std::endl;
      } else {
        std::cerr << "post " << sent_items << " " << action << " requests to host " << peer << std::endl;
        struct ::ibv_send_wr *bad;

        assert(q_front->wr.rdma.rkey != 0);
        int r = ::ibv_post_send(qp, q_front, &bad);
        check_ibv_status(r, bad);

        finish_completion_queue(sent_items, opcode);
      }

    } else {
      for (auto & p = q_front; p != NULL; p = p->next) {
        struct ::ibv_send_wr *bad;

        assert(q_front->wr.rdma.rkey != 0);
        ::ibv_send_wr *next = p->next;
        p->next = 0;
        int r = ::ibv_post_send(qp, p, &bad);
        check_ibv_status(r, bad);

        finish_completion_queue(1, opcode);
        p->next = next;
      }
    }
  }

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
  /**
   * FIXME TODO FIXME TODO FIXME TODO FIXME TODO FIXME TODO FIXME
   * I DO NOT understand. If I move this method into the .cc file,
   * the state of ib verbs is corrupted. It gives me a new, never-initialized
   * version of the device list each time ibv_get_device_list() is invoked.
   * This problem goes away when the method is defined in the .h file.
   * RFHH
   * FIXME TODO FIXME TODO FIXME TODO FIXME TODO FIXME TODO FIXME
   */
  VIRTUAL void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args) {
    CHECK_DEV_LIST();
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
      ;
    CHECK_DEV_LIST();

    po::variables_map vm;
    po::basic_command_line_parser<char> clp(args);
    clp.options(desc);
    po::store(clp.run(), vm);
    po::notify(vm);
    CHECK_DEV_LIST();

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

    t_poll_cq      = Timer("RDMA poll cq");
    t_read         = Timer("RDMA read");
    t_local_read   = Timer("     local read");
    t_post_read    = Timer("     post read");
    t_finish_read  = Timer("     finish read");
    t_write        = Timer("RDMA write");
    t_local_write  = Timer("     local write");
    t_post_write   = Timer("     post write");
    t_finish_write = Timer("     finish write");
    t_barrier      = Timer("RDMA barrier");

    init_networking();  // along the way, defines my_rank_ and num_servers_
    CHECK_DEV_LIST();

    if (rd_open(&res_, IBV_QPT_RC, post_send_chunk_, 0) != 0) {
      throw QPerfException("rd_open");
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

#ifndef DISABLE_NETWORKING
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
#endif

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

    wc_.resize(post_send_chunk_);
    if (false) {
      q_recv_front_.resize(num_servers_);
      q_send_front_.resize(num_servers_);
      ::size_t q_size = std::max(cache_buffer_.capacity(),
                                 write_buffer_.capacity());
      recv_wr_.resize(q_size);
      recv_sge_.resize(q_size);
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


  VIRTUAL void ReadKVRecords(std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode) {
    if (rw_mode != RW_MODE::READ_ONLY) {
      std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
    }

    t_read.start();

#ifndef DISABLE_INFINIBAND
    if (false) {
      // Fill the linked lists of WR requests
      q_recv_front_.assign(num_servers_, NULL);
      current_recv_req_ = 0;
      assert(recv_wr_.capacity() >= key.size());
      for (::size_t i = 0; i < key.size(); i++) {
        ::size_t owner = HostOf(key[i]);
        if (owner == my_rank_) {
          t_local_read.start();
          // Read directly, without RDMA
          cache[i] = value_.area_ + OffsetOf(key[i]);

          bytes_local_read += value_size_ * sizeof(ValueType);
          t_local_read.stop();

        } else {
          ValueType *target = cache_buffer_.get(value_size_);
          cache[i] = target;

          struct ::ibv_send_wr *wr = &recv_wr_[current_recv_req_];
          struct ::ibv_sge *sge = &recv_sge_[current_recv_req_];
          current_recv_req_++;

          wr->wr_id = 42; // Yes!
          wr->num_sge = 1;
          wr->sg_list = sge;
          sge->addr   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(target));
          sge->length = value_size_ * sizeof(ValueType);
          sge->lkey   = cache_.region_.mr->lkey;

          wr->opcode = IBV_WR_RDMA_READ;
          wr->send_flags = 0;
          wr->wr.rdma.remote_addr = peer_[owner].props.value +
            OffsetOf(key[i]) * sizeof(ValueType);
          wr->wr.rdma.rkey = peer_[owner].props.value_rkey;
          assert(wr->wr.rdma.rkey != 0);

          wr->next = q_recv_front_[owner];
          q_recv_front_[owner] = wr;

          bytes_remote_read += sge->length;
        }
      }

      // Post the linked lists of WR read requests
      for (::size_t i = 0; i < num_servers_; ++i) {
        post_linked_list(peer_[i].connection.qp, q_recv_front_[i], i,
                         IBV_WC_RDMA_READ);
      }

      // std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
      // Answer: no
    } else {
      std::vector<std::vector<const void *>> local_addr(num_servers_);
      std::vector<std::vector<const void *>> remote_addr(num_servers_);
      std::vector<std::vector< ::size_t>> sizes(num_servers_);
      std::vector< uint32_t> remote_key(num_servers_);
      for (::size_t i = 0; i < key.size(); i++) {
        ::size_t owner = HostOf(key[i]);
        if (owner == my_rank_) {
          t_local_read.start();
          // Read directly, without RDMA
          cache[i] = value_.area_ + OffsetOf(key[i]);

          bytes_local_read += value_size_ * sizeof(ValueType);
          t_local_read.stop();

        } else {
          ValueType *target = cache_buffer_.get(value_size_);
          cache[i] = target;

          local_addr[owner].push_back(target);
          sizes[owner].push_back(value_size_ * sizeof(ValueType));

          remote_addr[owner].push_back((const ValueType *)(peer_[owner].props.value +
            OffsetOf(key[i]) * sizeof(ValueType)));

          bytes_remote_read += value_size_ * sizeof(ValueType);
        }
      }

      for (::size_t h = 0; h < num_servers_; ++h) {
        uint32_t remote_key = peer_[h].props.value_rkey;

        if (local_addr[h].size() > 0) {
          ::size_t at = 0;
          ::size_t seen = 0;
          ::size_t n = post_send_chunk_;
          while (seen < local_addr[h].size()) {
            ::size_t p = std::min(n, local_addr[h].size() - at);
            if (p > 0) {
              t_post_read.start();
              if (rd_post_rdma_std(&peer_[h].connection,
                                   cache_.region_.mr->lkey,
                                   remote_key,
                                   IBV_WR_RDMA_READ,
                                   p,
                                   local_addr[h].data() + at,
                                   remote_addr[h].data() + at,
                                   sizes[h].data() + at) != 0) {
                throw QPerfException("rd_post_rdma_std");
              }
              t_post_read.stop();
              at += p;
            }

            t_finish_read.start();
            n = rd_poll(&res_, wc_.data(), wc_.size());
            t_finish_read.stop();
            for (::size_t i = 0; i < n; i++) {
              int status = wc_[i].status;
              if (status != IBV_WC_SUCCESS) {
                throw QPerfException("rd_poll");
              }
            }
            seen += n;
          }
        }
      }
    }

#endif
    t_read.stop();
  }


  VIRTUAL void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) {
    t_write.start();
#ifndef DISABLE_INFINIBAND
    // Fill the linked lists of SGE requests
    q_send_front_.assign(num_servers_, NULL);
    if (send_wr_.capacity() < key.size()) {
      send_wr_.reserve(key.size());
      send_sge_.reserve(key.size());
    }
    current_send_req_ = 0;
    for (::size_t i = 0; i < key.size(); i++) {
      ::size_t owner = HostOf(key[i]);
      if (owner == my_rank_) {
        t_local_write.start();
        // Write directly, without RDMA
        ValueType *target = value_.area_ + OffsetOf(key[i]);
        memcpy(target, value[i], value_size_ * sizeof(ValueType));
        t_local_write.stop();

        bytes_local_written += value_size_ * sizeof(ValueType);

      } else {
        assert(current_send_req_ < send_wr_.capacity());
        assert(current_send_req_ < recv_sge_.capacity());
        struct ::ibv_send_wr *wr = &send_wr_[current_send_req_];
        struct ::ibv_sge *sge = &recv_sge_[current_send_req_];
        current_send_req_++;

        memset(wr, 0, sizeof *wr);
        wr->num_sge = 1;
        wr->sg_list = sge;
        if (write_.contains(value[i])) {
          sge->addr = reinterpret_cast<uint64_t>(value[i]);
        } else {
          ValueType *v = write_buffer_.get(value_size_);
          memcpy(v, value[i], value_size_ * sizeof(ValueType));
          sge->addr = reinterpret_cast<uint64_t>(v);
        }
        sge->lkey = write_.region_.mr->lkey;
        sge->length = value_size_ * sizeof(ValueType);

        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr.rdma.remote_addr = peer_[owner].props.value +
          OffsetOf(key[i]) * sizeof(ValueType);
        wr->wr.rdma.rkey = peer_[owner].props.value_rkey;

        wr->next = q_send_front_[owner];
        q_send_front_[owner] = wr;

        bytes_remote_written += sge->length;
      }
    }

    // Post the linked lists of WR write requests
    for (::size_t i = 0; i < num_servers_; ++i) {
      post_linked_list(peer_[i].connection.qp, q_send_front_[i], i,
                       IBV_WC_RDMA_WRITE);
    }

    // std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
    // Answer: no
#endif
    t_write.stop();
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
    CHECK_DEV_LIST();
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
#ifndef DISABLE_NETWORKING
  void alltoall_leaf(const char *sendbuf, ::size_t send_item_size,
                     char *recvbuf, ::size_t recv_item_size,
                     ::size_t me, ::size_t size,
                     ::size_t start, ::size_t size_2pow);
  void alltoall_DC(const char *sendbuf, ::size_t send_item_size,
                   char *recvbuf, ::size_t recv_item_size,
                   ::size_t me, ::size_t size,
                   ::size_t start, ::size_t size_2pow);
#endif
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
      t_poll_cq.start();
      do {

        r = ::ibv_poll_cq(res_.cq, n, wc_.data() + seen);
        if (r < 0) {
          std::cerr << "Oops, must handle wrong result cq poll" << std::endl;
          break;
        }
      } while (r == 0);
      t_poll_cq.stop();
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

#if 0       // DEPRECATED
/******************************************************************************
 * Function: connect_qp
 *
 * Input
 * none
 *
 * Output
 * none
 *
 * Returns
 * none
 *
 * @throws RDMAException
 *
 * Description
 * Connect the QP. Transition the server side to RTR, sender side to RTS
 ******************************************************************************/
  void connect_qp() {

    /* exchange using o-o-b network info required to connect QPs */
    std::vector<cm_con_data_t> my_con_data(num_servers_);
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        cm_con_data_t *con = &my_con_data[i];
        con->value  = reinterpret_cast<uint64_t>(value_.vaddr);
        con->cache  = reinterpret_cast<uint64_t>(cache_.vaddr);
        con->value_rkey   = value_.key;
        con->cache_rkey   = cache_.key;
        con->qp_num = peer_[i].connection.qp->qp_num;
        con->lid    = res_.lnode.lid;
#if 0
        con->gid    = my_gid;
#endif
        fprintf (stdout, "\nLocal LID[%zd] = 0x%x\n", i, res_.lnode.lid);
      }
    }

    std::vector<cm_con_data_t> remote_con_data(num_servers_);
    alltoall(my_con_data.data(), sizeof my_con_data[0],
             remote_con_data.data(), sizeof remote_con_data[0]);

    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        /* save the remote side attributes, we will need it for the post SR */
        peer_[i].props = remote_con_data[i];

        std::cout << "Peer " << i << std::endl;
        std::cout << remote_con_data[i];
#if 0
        if (config_.gid_idx >= 0) {
          std::cout << "  gid " << gid_t(peer_[i].props.gid);
        }
#endif
        std::cout << std::endl << std::dec;
      }
    }

    barrier();
  }
#endif

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

#else   // def USE_MPI

  void init_networking() {
#ifndef DISABLE_NETWORKING
    std::string network("socket");
    // std::string interface("ib0");
    std::string interface("br0");
    std::cerr << "For now, hardwire network = " << network << ", interface = " << interface << std::endl;

    mr::OptionList options;
    options.push_back(mr::Option("interface", interface));
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
#else
    std::cerr << "Skip the network thingy" << std::endl;
    const char *prun_cpu_rank = getenv("PRUN_CPU_RANK");
    const char *nhosts = getenv("NHOSTS");
    my_rank_ = atoi(prun_cpu_rank);
    num_servers_ = atoi(nhosts);
#endif
  }


#ifndef DISABLE_NETWORKING
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
#endif


 public:
  void alltoall(const void *sendbuf, ::size_t send_item_size,
                              void *recvbuf, ::size_t recv_item_size) {
    CHECK_DEV_LIST();
#ifndef DISABLE_NETWORKING
    ::size_t size_2pow = mr::next_2_power(num_servers_);

    alltoall_DC(static_cast<const char *>(sendbuf), send_item_size,
                static_cast<char *>(recvbuf), recv_item_size,
                my_rank_, num_servers_, 0, size_2pow);
#endif

    // copy our own contribution by hand
    memcpy(static_cast<char *>(recvbuf) + my_rank_ * recv_item_size,
           static_cast<const char *>(sendbuf) + my_rank_ * send_item_size,
           recv_item_size);
    CHECK_DEV_LIST();
  }


  void barrier() {
    t_barrier.start();
#ifndef DISABLE_NETWORKING
    int32_t dummy = -1;
    if (my_rank_ == 0) {
      broadcast_.bcast_send(dummy);
      dummy = broadcast_.reduce_rcve<int32_t>();
    } else {
      broadcast_.bcast_rcve(&dummy);
      broadcast_.reduce_send(dummy);
    }
#endif
    t_barrier.stop();
  }

#endif  // def USE_MPI
#endif  // ndef OK_TO_USE_CC_FILE

 private:
  ::size_t num_servers_;
  ::size_t my_rank_;
  
  DEVICE res_;
  std::vector<rdma_peer> peer_;

  std::vector< ::ibv_send_wr> send_wr_;
  std::vector< ::ibv_sge> send_sge_;
  ::size_t current_send_req_;
  std::vector< ::ibv_send_wr *> q_send_front_;
  std::vector< ::ibv_send_wr> recv_wr_;
  std::vector< ::ibv_sge> recv_sge_;
  ::size_t current_recv_req_;
  std::vector< ::ibv_send_wr *> q_recv_front_;
  std::vector< ::ibv_wc> wc_;

  std::string dev_name_;
  ::size_t post_send_chunk_ = 1024;
  const bool REQUIRE_POSTED_RECEIVE;

  /* memory buffer pointers, used for RDMA and send ops */
  rdma_area<ValueType> value_;
  rdma_area<ValueType> cache_;
  rdma_area<ValueType> write_;

#ifndef USE_MPI
#ifndef DISABLE_NETWORKING
  mr::net::Network *network_;
  mr::net::Broadcast broadcast_;
  mr::net::Network::RWList rw_list_;
#endif
#endif

  Timer t_poll_cq;
  Timer t_read;
  Timer t_local_read;
  Timer t_post_read;
  Timer t_finish_read;
  Timer t_write;
  Timer t_local_write;
  Timer t_post_write;
  Timer t_finish_write;
  Timer t_barrier;

  int64_t bytes_local_read = 0;
  int64_t bytes_remote_read = 0;
  int64_t bytes_local_written = 0;
  int64_t bytes_remote_written = 0;
};

}   // namespace DKVRDMA
}   // namespace DKV

#endif  // ndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
