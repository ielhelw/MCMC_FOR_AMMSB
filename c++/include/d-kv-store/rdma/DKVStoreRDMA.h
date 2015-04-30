/*
 * Copyright
 */
#ifndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
#define APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__

// #define OK_TO_DEFINE_IN_CC_FILE

#include <errno.h>
#include <string.h>
#include <inttypes.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <infiniband/verbs.h>

#ifndef USE_MPI
#ifndef DISABLE_NETWORKING
#include <mr/net/netio.h>
#include <mr/net/broadcast.h>
#endif
#endif

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

#include <d-kv-store/DKVStore.h>

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

extern struct ibv_device **global_dev_list;

#define CHECK_DEV_LIST() \
  do { \
    int num; \
    std::cerr << __FILE__ << "." << __LINE__ << \
         ": dev_list[0] " << (void *)::DKV::DKVRDMA::global_dev_list[0] << \
         " ibv_get_device_list()[0] " << ::ibv_get_device_list(&num)[0] << \
         " " << ::DKV::DKVRDMA::global_dev_list[0]->dev_name << std::endl; \
    /* check_dev_list(__FILE__, __LINE__); */ \
  } while (0)

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


class NetworkException : public RDMAException {
 public:
  NetworkException(const std::string &reason) throw() : RDMAException(reason) {
  }

  virtual ~NetworkException() throw() {
  }
};


struct config_t {
  config_t() : dev_name(""), ib_port(1), gid_idx(-1) {
  }

  void setName(const char *name) {
    dev_name = std::string(name);
  }

  std::ostream &put(std::ostream &s) const {
    s << " ------------------------------------------------" << std::endl;
    s << " Device name : " << dev_name << std::endl;
    s << " IB port : " << ib_port << std::endl;
    if (gid_idx >= 0)
      s << " GID index : " << gid_idx << std::endl;
    s << " ------------------------------------------------" << std::endl <<
      std::endl;
    return s;
  }

  std::string dev_name; /* IB device name */
  int32_t ib_port;      /* local IB port to work with */
  int32_t gid_idx;      /* gid index to use */
};

inline std::ostream &operator<< (std::ostream &s, const config_t &c) {
  return c.put(s);
}


struct gid_t {
  gid_t(union ::ibv_gid gid) : gid(gid) {
  }

  std::ostream &put(std::ostream &s) const {
    std::ios_base::fmtflags flags = s.flags();
    s << std::hex << std::setw(2) << std::setfill('0');

    ::size_t upb = sizeof gid.raw / sizeof gid.raw[0];
    for (::size_t i = 0; i < upb - 1; i++) {
      s << "0x" << gid.raw[i] << ":";
    }
    s << "0x" << gid.raw[upb - 1];

    s.flags(flags);

    return s;
  }

  union ::ibv_gid gid;
};

inline std::ostream &operator<< (std::ostream &s, const gid_t &g) {
  return g.put(s);
}


/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t {
  uint64_t value;       /* Buffer address */
  uint64_t cache;       /* Buffer address */
  uint32_t value_rkey;  /* Remote key */
  uint32_t cache_rkey;  /* Remote key */
  uint32_t qp_num;      /* QP number */
  uint16_t lid;         /* LID of the IB port */
  union ::ibv_gid gid;    /* gid */

  std::ostream &put(std::ostream &s) const {
    std::ios_base::fmtflags flags = s.flags();
    s << std::hex;
    s << "  Remote value address = 0x" << value << std::endl;
    s << "  Remote value rkey = 0x" << value_rkey << std::endl;
    s << "  Remote cache address = 0x" << cache << std::endl;
    s << "  Remote cache rkey = 0x" << cache_rkey << std::endl;
    s << "  Remote QP number = 0x" << qp_num << std::endl;
    s << "  Remote LID = 0x" << lid << std::endl;
    s.flags(flags);

    return s;
  }
} // __attribute__ ((packed))
;

inline std::ostream &operator<< (std::ostream &s, const cm_con_data_t &r) {
  return r.put(s);
}


struct rdma_peer {
  struct ::ibv_qp *qp;                  /* QP handle */
  struct ::ibv_port_attr port_attr;     /* IB port attributes */
  struct cm_con_data_t props;         /* values to connect to remote side */
  cm_con_data_t con_data;
};


template <typename ValueType>
class rdma_area {
 public:
  rdma_area() : n_elements_(0), mr_(NULL), area_(NULL) {
  }

  ~rdma_area() {
    if (mr_ != NULL) {
      if (::ibv_dereg_mr(mr_) != 0) {
        std::cerr << "failed to deregister MR" << std::endl;
      }
      mr_ = NULL;
    }
    if (mine_) {
      delete[] area_;
    }
  }

  void Init(::ibv_pd *pd, ::size_t n_elements, int mr_flags) {
    n_elements_ = n_elements;
    mr_flags_   = mr_flags;

    /* allocate the memory buffer that will hold the data */
    area_ = new ValueType[n_elements_];
    if (area_ == NULL) {
      throw RDMAException("cannot allocate value area");
    }

    mine_ = true;

    register_mr(pd);
  }

  void Init(::ibv_pd *pd, const Buffer<ValueType> &buffer, int mr_flags) {
    area_ = buffer.buffer();
    n_elements_ = buffer.capacity();
    mr_flags_ = mr_flags;

    mine_ = false;

    register_mr(pd);
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
    s << ", lkey=0x" << mr_->lkey;
    s << ", rkey=0x" << mr_->rkey;
    s << ", flags=0x" << mr_flags_;
    s << ", mr=" << (void *)mr_;
    s.flags(flags);

    return s;
  }

 private:
  /* register the memory */
  void register_mr(::ibv_pd *pd) {
    mr_ = ::ibv_reg_mr(pd, area_, n_elements_ * sizeof(ValueType), mr_flags_);
    if (mr_ == NULL) {
      throw RDMAException(errno, "ibv_reg_mr failed with mr_flags=0x" +
                          to_string(mr_flags_));
    }
  }

 public:
  ::size_t n_elements_;
  int mr_flags_;
  struct ::ibv_mr *mr_;                  /* MR handle for buf */
  ValueType *area_;

 private:
  bool mine_;
};

template <typename ValueType>
inline std::ostream &operator<< (std::ostream &s,
                                 const rdma_area<ValueType> &r) {
  return r.put(s);
}


/* structure of system resources */
template <typename ValueType>
struct resources {

/* structure of system resources */
  resources()
      : ib_ctx_(NULL), pd_(NULL), cq_(NULL) {
    CHECK_DEV_LIST();
    ibv_dev_list_ = ::ibv_get_device_list(&ibv_num_devices_);
std::cerr << __FILE__ << "." << __LINE__ << ": cached dev_list[0] " << ibv_dev_list_[0] << std::endl;
  }

  ~resources() {
    std::cerr << "******************************* NOOOOOOOOOOO ******************" << std::cerr;
    for (auto & p : peer_) {
      if (p.qp != NULL) {
        if (::ibv_destroy_qp(p.qp) != 0) {
          std::cerr << "failed to destroy QP" << std::endl;
        }
      }
    }
    if (cq_ != NULL) {
      if (::ibv_destroy_cq (cq_) != 0) {
        std::cerr << "failed to destroy CQ" << std::endl;
      }
    }
    if (pd_ != NULL) {
      if (::ibv_dealloc_pd (pd_) != 0) {
        std::cerr << "failed to deallocate PD" << std::endl;
      }
    }
    if (ib_ctx_ != NULL) {
      if (::ibv_close_device (ib_ctx_) != 0) {
        std::cerr << "failed to close device context" << std::endl;
      }
    }
  }

  void Init(config_t *config, ::size_t my_rank, ::size_t num_servers,
            ::size_t max_requests) {

CHECK_DEV_LIST();
    std::cerr << "RDMA: searching for IB devices in host" << std::endl;
    /* get device names in the system */
    // int num_devices;
    // struct ::ibv_device **dev_list = ::ibv_get_device_list(&num_devices);
    int num_devices = ibv_num_devices_;
    struct ::ibv_device **dev_list = ibv_dev_list_;
    if (!dev_list) {
      throw RDMAException(errno, "failed to get IB devices list");
    }
CHECK_DEV_LIST();
    /* if there isn't any IB device in host */
    if (num_devices == 0) {
      throw RDMAException(errno, "found no IB devices");
    }
    std::cerr << "RDMA: found " << num_devices << " device(s)" << std::endl;
CHECK_DEV_LIST();

    /* search for the specific device we want to work with */
struct ::ibv_device *ib_dev = NULL;
    for (auto i = 0; i < num_devices; i++) {
      if (config->dev_name == "") {
        ib_dev = dev_list[i];
if (0) {
        config->setName(::ibv_get_device_name(dev_list[i]));
        std::cerr << "device not specified, using first one found: " <<
          ::ibv_get_device_name(dev_list[i]) << " save as " <<
          config->dev_name << std::endl;
}
        break;
      } else if (::ibv_get_device_name(dev_list[i]) == config->dev_name) {
        ib_dev = dev_list[i];
        break;
      }
    }
    /* if the device wasn't found in host */
    if (ib_dev == NULL) {
      throw RDMAException(errno, "IB device " + config->dev_name + " not found");
    }

CHECK_DEV_LIST();
    std::cerr << "dev_list " << (void *)dev_list << std::endl;
    std::cerr << "dev_list[0] " << (void *)dev_list[0] << std::endl;
    std::cerr << "ib_dev " << (void *)ib_dev << " ->dev_name " << ib_dev->dev_name << std::endl;
CHECK_DEV_LIST();
    /* get device handle */
    ib_ctx_ = ::ibv_open_device(ib_dev);
    if (ib_ctx_ == NULL) {
      throw RDMAException(errno, "failed to open device " + config->dev_name);
    }
    std::cerr << __LINE__ << ": ib_ctx_->ops.post_recv " << (void *)ib_ctx_->ops.post_recv << std::endl;

    /* We are now done with device list, free it */
    ::ibv_free_device_list(dev_list);
    dev_list = NULL;
    ib_dev = NULL;

    peer_.resize(num_servers);
    /* query port properties */
    for (::size_t i = 0; i < num_servers; i++) {
      if (i == my_rank) {
        std::cerr << "FIXME ibv_query_port " << __LINE__ << ": What about me?" << std::endl;
      } else {
        std::cerr << __LINE__ << ": config->ib_port " << config->ib_port << std::endl;
        memset(&peer_[i].port_attr, 0, sizeof peer_[i].port_attr);
        if (::ibv_query_port (ib_ctx_, config->ib_port, &peer_[i].port_attr)) {
          throw RDMAException(errno, "ibv_query_port on port " +
                              to_string(config->ib_port) + " failed");
        }
      }
    }

    /* allocate Protection Domain */
    pd_ = ::ibv_alloc_pd(ib_ctx_);
    if (pd_ == NULL) {
      throw RDMAException(errno, "ibv_alloc_pd failed");
    }

    cq_ = ::ibv_create_cq (ib_ctx_, max_requests, NULL, NULL, 0);
    if (cq_ == NULL) {
      throw RDMAException(errno, "failed to create CQ with " +
                          to_string(num_servers) + "entries");
    }

    /* create the Queue Pair */
    for (::size_t i = 0; i < num_servers; ++i) {
      if (i == my_rank) {
        std::cerr << "FIXME QP " << __LINE__ << ": What about me?" << std::endl;
        peer_[i].qp = NULL;
      } else {
        struct ::ibv_qp_init_attr qp_init_attr;
        memset (&qp_init_attr, 0, sizeof qp_init_attr);
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.sq_sig_all = 1;
        qp_init_attr.send_cq = cq_;
        qp_init_attr.recv_cq = cq_;
        qp_init_attr.cap.max_send_wr = 1;
        qp_init_attr.cap.max_recv_wr = 1;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        peer_[i].qp = ::ibv_create_qp (pd_, &qp_init_attr);
        if (peer_[i].qp == NULL) {
          throw RDMAException(errno, "failed to create QP");
        }
        std::cout << "QP[" << i << "] was created, QP number=0x" <<
          std::hex << peer_[i].qp->qp_num << std::endl << std::dec;
        std::cerr << __LINE__ << ": qp->context->ops.post_recv " << (void *)peer_[i].qp->context->ops.post_recv << std::endl;
      }
    }
  }

  struct ::ibv_pd *pd() {
    return pd_;
  }

 private:
  int ibv_num_devices_;
  struct ::ibv_device **ibv_dev_list_;

  struct ::ibv_context *ib_ctx_;         /* device handle */
  struct ::ibv_pd *pd_;                  /* PD handle */
  struct ::ibv_cq *cq_;                  /* CQ handle */

  std::vector<rdma_peer> peer_;       /* state of connection to peer_[i] */

  friend class DKVStoreRDMA;
};


struct q_item_t {
  q_item_t() {
    wr.sg_list = &sg_list;
  }

  struct ::ibv_send_wr    wr;
  struct ::ibv_sge        sg_list;
  q_item_t             *next;
};


/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  DKVStoreRDMA() : DKVStoreInterface() {
CHECK_DEV_LIST();
CHECK_DEV_LIST();
  }

  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

#ifndef OK_TO_DEFINE_IN_CC_FILE
  virtual ~DKVStoreRDMA() {
#ifndef DISABLE_NETWORKING
#  ifndef USE_MPI
    delete network_;
#  endif
#endif
  }
#else
  virtual ~DKVStoreRDMA();
#endif

  virtual void InfoH() const {
    CHECK_DEV_LIST();
  }

  virtual void Info() const;

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
  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args) {
    CHECK_DEV_LIST();
    namespace po = ::boost::program_options;

    po::options_description desc("RDMA options");
    desc.add_options()
      // ("rdma.oob-port", po::value(&config_.tcp_port)->default_value(0), "RDMA out-of-band TCP port")
      ("rdma:dev",  po::value(&config_.dev_name), "RDMA device name")
      ("rdma:port", po::value(&config_.ib_port),  "RDMA device port")
      ("rdma:gid",  po::value(&config_.gid_idx),  "RDMA GID index")
      ;
    CHECK_DEV_LIST();

    po::variables_map vm;
    po::basic_command_line_parser<char> clp(args);
    clp.options(desc);
    po::store(clp.run(), vm);
    po::notify(vm);
    CHECK_DEV_LIST();

    /* print the used parameters for info*/
    std::cout << config_;

    init_networking();  // along the way, defines my_rank_ and num_servers_
    CHECK_DEV_LIST();

#ifndef DISABLE_NETWORKING
    /* init all of the resources, so cleanup will be easy */
    /* create resources before using them */
    res_.Init(&config_, my_rank_, num_servers_,
              std::max(max_cache_capacity, max_write_capacity));
#endif

    ::DKV::DKVStoreInterface::Init(value_size, total_values,
                                   max_cache_capacity, max_write_capacity,
                                   args);
    /* memory buffer to hold the value data */
    ::size_t my_values = (total_values + num_servers_ - 1) / num_servers_;
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    value_.Init(res_.pd(), my_values * value_size, mr_flags);
    std::cout << "MR/value " << value_ << std::endl;

    /* memory buffer to hold the cache data */
    mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE;
    cache_.Init(res_.pd(), cache_buffer_, mr_flags);
    std::cout << "MR/cache " << cache_ << std::endl;

    /* memory buffer to hold the zerocopy write data */
    mr_flags = IBV_ACCESS_LOCAL_WRITE;
    write_.Init(res_.pd(), write_buffer_, mr_flags);
    std::cout << "MR/write " << write_ << std::endl;

#ifndef DISABLE_NETWORKING
    /* connect the QPs */
    connect_qp();
#endif

    q_recv_front_.resize(num_servers_);
    q_send_front_.resize(num_servers_);
    ::size_t q_size = std::max(cache_buffer_.capacity(),
                               write_buffer_.capacity());
    recv_wr_.resize(q_size);
    recv_sge_.resize(q_size);
  }


  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode) {
    if (rw_mode != RW_MODE::READ_ONLY) {
      std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
    }

#ifndef DISABLE_NETWORKING
    // Fill the linked lists of WR requests
    q_recv_front_.assign(num_servers_, NULL);
    current_recv_req_ = 0;
    assert(recv_wr_.capacity() >= key.size());
    for (::size_t i = 0; i < key.size(); i++) {
      ::size_t owner = HostOf(key[i]);
      if (owner == my_rank_) {
        // Read directly, without RDMA
        cache[i] = value_.area_ + OffsetOf(key[i]);

      } else {
        ValueType *target = cache_buffer_.get(value_size_);
        cache[i] = target;

        struct ::ibv_send_wr *wr = &recv_wr_[current_recv_req_];
        struct ::ibv_sge *sge = &recv_sge_[current_recv_req_];
        current_recv_req_++;

        wr->num_sge = 1;
        wr->sg_list = sge;
        sge->addr   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(target));
        sge->length = value_size_ * sizeof(ValueType);
        sge->lkey   = cache_.mr_->lkey;

        wr->opcode = IBV_WR_RDMA_READ;
        wr->send_flags = 0;
        wr->wr.rdma.remote_addr = res_.peer_[owner].props.value +
          OffsetOf(key[i]) * sizeof(ValueType);
        wr->wr.rdma.rkey = res_.peer_[owner].props.value_rkey;
        assert(wr->wr.rdma.rkey != 0);

        wr->next = q_recv_front_[owner];
        q_recv_front_[owner] = wr;
      }
    }

    // Post the linked list of WR requests
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        // already done the memcpy
        std::cerr << "Skip the home reads" << std::endl;
      } else {
        struct ::ibv_send_wr *bad;

        ::size_t sent_items = 0;
        for (auto r = q_recv_front_[i]; r != NULL; r = r->next) {
          sent_items++;
        }

        if (sent_items == 0) {
          std::cerr << "skip posting " << sent_items << " read requests to host " << i << std::endl;
        } else {
          std::cerr << "post " << sent_items << " read requests to host " << i << std::endl;

          assert(q_recv_front_[i]->wr.rdma.rkey != 0);
          int r = ::ibv_post_send(res_.peer_[i].qp, q_recv_front_[i], &bad);
          check_ibv_status(r, bad);

          finish_completion_queue(sent_items, IBV_WC_RDMA_READ);
        }
      }
    }

    std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
#endif
  }


  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) {
#ifndef DISABLE_NETWORKING
    // Fill the linked lists of SGE requests
    q_send_front_.assign(num_servers_, NULL);
    std::cerr << "FIXME: statically determine UPB for send items" << std::endl;
    if (send_wr_.capacity() < key.size()) {
      send_wr_.reserve(key.size());
      send_sge_.reserve(key.size());
    }
    current_send_req_ = 0;
    for (::size_t i = 0; i < key.size(); i++) {
      ::size_t owner = HostOf(key[i]);
      if (owner == my_rank_) {
        // Write directly, without RDMA
        ValueType *target = value_.area_ + OffsetOf(key[i]);
        memcpy(target, value[i], value_size_ * sizeof(ValueType));

      } else {
        assert(current_send_req_ < send_wr_.capacity());
        assert(current_send_req_ < recv_sge_.capacity());
        struct ::ibv_send_wr *wr = &send_wr_[current_send_req_];
        struct ::ibv_sge *sge = &recv_sge_[current_send_req_];
        current_send_req_++;

        wr->num_sge = 1;
        wr->sg_list = sge;
        if (write_.contains(value[i])) {
          sge->addr = reinterpret_cast<uint64_t>(value[i]);
        } else {
          ValueType *v = write_buffer_.get(value_size_);
          memcpy(v, value[i], value_size_ * sizeof(ValueType));
          sge->addr = reinterpret_cast<uint64_t>(v);
        }
        sge->lkey = write_.mr_->lkey;
        sge->length = value_size_ * sizeof(ValueType);

        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr.rdma.remote_addr = res_.peer_[owner].props.value +
          OffsetOf(key[i]) * sizeof(ValueType);
        wr->wr.rdma.rkey = res_.peer_[owner].props.value_rkey;

        wr->next = q_send_front_[owner];
        q_send_front_[owner] = wr;
      }
    }

    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        // already done the memcpy
        std::cerr << "Skip the home writes" << std::endl;
      } else {
        struct ::ibv_send_wr *bad;

        ::size_t sent_items = 0;
        for (auto r = q_send_front_[i]; r != NULL; r = r->next) {
          sent_items++;
        }
        if (sent_items == 0) {
          std::cerr << "skip posting " << sent_items << " write requests to host " << i << std::endl;
        } else {
          std::cerr << "post " << sent_items << " write requests to host " << i << std::endl;

          int r = ::ibv_post_send(res_.peer_[i].qp, q_send_front_[i], &bad);
          check_ibv_status(r, bad);

          finish_completion_queue(sent_items, IBV_WC_RDMA_WRITE);
        }
      }
    }

    std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
#endif
  }


  virtual std::vector<ValueType *> GetWriteKVRecords(::size_t n) {
    std::vector<ValueType *> w(n);
    for (::size_t i = 0; i < n; i++) {
      w[i] = write_buffer_.get(value_size_);
    }

    return w;
  }


  virtual void FlushKVRecords(const std::vector<KeyType> &key) {
    std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
    write_buffer_.reset();
  }

  /**
   * Purge the cache area
   */
  virtual void PurgeKVRecords() {
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
  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args);

  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  virtual void FlushKVRecords(const std::vector<KeyType> &key);

  std::vector<ValueType *> GetWriteKVRecords(::size_t n);

  /**
   * Purge the cache area
   */
  virtual void PurgeKVRecords();

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
#ifndef DISABLE_NETWORKING
    std::cerr << "FIXME -- lift this vector to be static" << std::endl;
    std::vector<struct ::ibv_wc> wc(n);
    int r;
    ::size_t seen = 0;
    while (seen < n) {
      do {
        r = ::ibv_poll_cq(res_.cq_, wc.size(), wc.data());
        if (r < 0) {
          std::cerr << "Oops, must handle wrong result cq poll" << std::endl;
          break;
        }
      } while (r == 0);
      for (int i = 0; i < r; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          throw RDMAException("ibv_poll_cq status " + to_string(wc[r].status) +
                              " " +
                              std::string(ibv_wc_status_str(wc[r].status)));
        }
        assert(wc[i].opcode == opcode);
      }
      seen += r;
    }
#endif
  }

/******************************************************************************
 * Function: modify_qp_to_init
 *
 * Input
 * qp QP to transition
 *
 * Output
 * none
 *
 * @throws
 * RDMAException(::ibv_modify_qp) failure
 *
 * Description
 * Transition a QP from the RESET to INIT state
 ******************************************************************************/
  void modify_qp_to_init (struct ::ibv_qp *qp) {
    struct ::ibv_qp_attr attr;
    int flags;
    memset (&attr, 0, sizeof (attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = config_.ib_port;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
      IBV_QP_ACCESS_FLAGS;
    int rc = ::ibv_modify_qp (qp, &attr, flags);
    if (rc != 0) {
      throw RDMAException(rc, "failed to modify QP state to INIT");
    }
  }

/******************************************************************************
 * Function: modify_qp_to_rtr
 *
 * Input
 * qp QP to transition
 * remote_qpn remote QP number
 * dlid destination LID
 * dgid destination GID (mandatory for RoCEE)
 *
 * Output
 * none
 *
 * @throws
 * RDMAException(::ibv_modify_qp) failure
 *
 * Description
 * Transition a QP from the INIT to RTR state, using the specified QP number
 ******************************************************************************/
  void modify_qp_to_rtr(struct ::ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
                        const gid_t &dgid) {
    struct ::ibv_qp_attr attr;
    int flags;
    memset (&attr, 0, sizeof (attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_256;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = config_.ib_port;
    if (config_.gid_idx >= 0) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.port_num = 1;
      memcpy (&attr.ah_attr.grh.dgid, &dgid.gid, 16);
      attr.ah_attr.grh.flow_label = 0;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.grh.sgid_index = config_.gid_idx;
      attr.ah_attr.grh.traffic_class = 0;
    }
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    int rc = ::ibv_modify_qp (qp, &attr, flags);
    if (rc != 0) {
      throw RDMAException(rc, "failed to modify QP state to RTR");
    }
  }

/******************************************************************************
 * Function: modify_qp_to_rts
 *
 * Input
 * qp QP to transition
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, ibv_modify_qp failure code on failure
 *
 * Description
 * Transition a QP from the RTR to RTS state
 ******************************************************************************/
  void modify_qp_to_rts (struct ::ibv_qp *qp) {
    struct ::ibv_qp_attr attr;
    int flags;
    memset (&attr, 0, sizeof (attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 6;
    attr.rnr_retry = 0;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    int  rc = ::ibv_modify_qp (qp, &attr, flags);
    if (rc != 0) {
      throw RDMAException(rc, "failed to modify QP state to RTS");
    }
  }

/******************************************************************************
 * Function: post_receive
 *
 * Input
 * area pointer to resources structure
 *
 * Output
 * none
 *
 * @throws RDMAException
 *
 * Description
 *
 ******************************************************************************/
  void post_receive(struct rdma_area<ValueType> *area, struct ::ibv_qp *qp) {
    struct ::ibv_recv_wr rr;
    struct ::ibv_sge sge;
    struct ::ibv_recv_wr *bad_wr = NULL;
    int rc;
    /* prepare the scatter/gather entry */
    memset (&sge, 0, sizeof (sge));
    sge.addr = reinterpret_cast<uintptr_t>(area->area_);
    sge.length = area->n_elements_ * value_size_ * sizeof *area->area_;
    sge.lkey = area->mr_->lkey;
    /* prepare the receive work request */
    memset(&rr, 0, sizeof (rr));
    rr.next = NULL;
    rr.wr_id = 0;
    rr.sg_list = &sge;
    rr.num_sge = 1;
    /* post the Receive Request to the RQ */
    rc = ::ibv_post_recv(qp, &rr, &bad_wr);
    if (rc != 0) {
      throw RDMAException(rc, "failed to post RR");
    }
  }

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
    int rc = 0;
    union ::ibv_gid my_gid;
    if (config_.gid_idx >= 0) {
      rc = ::ibv_query_gid (res_.ib_ctx_, config_.ib_port, config_.gid_idx, &my_gid);
      if (rc != 0) {
        throw RDMAException(rc, "could not get gid for port " +
                            to_string(config_.ib_port) + ", index " +
                            to_string(config_.gid_idx));
      }
    } else {
      memset (&my_gid, 0, sizeof my_gid);
    }

    /* exchange using o-o-b network info required to connect QPs */
    std::vector<cm_con_data_t> my_con_data(num_servers_);
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        cm_con_data_t *con = &my_con_data[i];
        con->value  = (uint64_t) value_.area_;
        con->cache  = (uint64_t) cache_.area_;
        con->value_rkey   = value_.mr_->rkey;
        con->cache_rkey   = cache_.mr_->rkey;
        con->qp_num = res_.peer_[i].qp->qp_num;
        con->lid    = res_.peer_[i].port_attr.lid;
        con->gid    = my_gid;
        fprintf (stdout, "\nLocal LID[%zd] = 0x%x\n", i, res_.peer_[i].port_attr.lid);
      }
    }

    std::vector<cm_con_data_t> remote_con_data(num_servers_);
    alltoall(my_con_data.data(), sizeof my_con_data[0],
             remote_con_data.data(), sizeof remote_con_data[0]);

    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        res_.peer_[i].con_data = remote_con_data[i];
        /* save the remote side attributes, we will need it for the post SR */
        res_.peer_[i].props = remote_con_data[i];

        std::cout << "Peer " << i << std::endl;
        std::cout << remote_con_data[i];
        if (config_.gid_idx >= 0) {
          std::cout << "  gid " << gid_t(res_.peer_[i].con_data.gid) << std::endl;
        }
        std::cout << std::dec;
      }
    }

    /* modify the QP to init */
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        modify_qp_to_init(res_.peer_[i].qp);
      }
    }

    // FIXME: What should we do about this? Do we _ever_ receive?
    /* let the client post RR to be prepared for incoming messages */
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        // post_receive(&value_, res_.peer_[i].qp);
        post_receive(&cache_, res_.peer_[i].qp);
      }
    }
    fprintf (stderr, "Posted receive to QP\n");

    std::cerr << "FIXME: do we need a barrier here?" << std::endl;

    /* modify the QP to RTR */
    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        modify_qp_to_rtr(res_.peer_[i].qp,
                         res_.peer_[i].con_data.qp_num,
                         res_.peer_[i].con_data.lid,
                         res_.peer_[i].con_data.gid);
      }
    }
    fprintf (stderr, "Modified QP state to RTR\n");

    std::cerr << "FIXME: do we need a barrier here?" << std::endl;

    for (::size_t i = 0; i < num_servers_; ++i) {
      if (i == my_rank_) {
        std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
      } else {
        modify_qp_to_rts(res_.peer_[i].qp);
      }
    }
    fprintf (stderr, "QP state was change to RTS\n");

    /* sync to make sure that both sides are in states that they can connect to prevent packet loose */
    barrier();
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

#else

  void init_networking() {
#ifndef DISABLE_NETWORKING
    std::string network("socket");
    // std::string interface("ib0");
    std::string interface("br0");
    std::cerr << "For now, hardwire network = " << network << ", interface = " << interface << std::endl;

    mr::OptionList options;
    options.push_back(mr::Option("interface", interface));
    network_ = mr::net::NetworkImpl::createNetwork("socket", options);
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
#endif


#ifndef DISABLE_NETWORKING
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
#ifndef DISABLE_NETWORKING
    int32_t dummy;
    if (my_rank_ == 0) {
      broadcast_.bcast_send(dummy);
      dummy = broadcast_.reduce_rcve<int32_t>();
    } else {
      broadcast_.bcast_rcve(&dummy);
      broadcast_.reduce_send(dummy);
    }
#endif
  }

#endif
#endif

 private:
  ::size_t num_servers_;
  ::size_t my_rank_;
  
  config_t config_;
  resources<ValueType> res_;

  std::vector<::ibv_send_wr> send_wr_;
  std::vector<::ibv_sge> send_sge_;
  ::size_t current_send_req_;
  std::vector<::ibv_send_wr *> q_send_front_;
  std::vector<::ibv_send_wr> recv_wr_;
  std::vector<::ibv_sge> recv_sge_;
  ::size_t current_recv_req_;
  std::vector<::ibv_send_wr *> q_recv_front_;

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
};

}   // namespace DKVRDMA
}   // namespace DKV

#endif  // ndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
