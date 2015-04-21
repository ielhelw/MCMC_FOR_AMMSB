/*
 * Copyright
 */
#ifndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
#define APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__

#include <string.h>
#include <inttypes.h>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include <infiniband/verbs.h>

#include <d-kv-store/DKVStore.h>

namespace DKV {
namespace DKVRDMA {

class RDMAException : public std::exception {
 public:
  RDMAException(const std::string &reason) throw() : reason_(reason) {
  }

  virtual ~RDMAException() throw() {
  }

  virtual const char *what() const throw() {
    return reason_.c_str();
  }
  
 protected:
  RDMAException() throw() : reason_("<apparently inherited>") {
  }

  std::string reason_;
};

class NetworkException : public RDMAException {
 public:
  NetworkException(const std::string &reason) throw() : RDMAException(reason) {
  }

  virtual ~NetworkException() {
  }
};


struct config_t {
  config_t();

  std::ostream &put(std::ostream &s) const;

  std::string dev_name; /* IB device name */
  int ib_port;          /* local IB port to work with */
  int gid_idx;          /* gid index to use */
};

inline std::ostream &operator<< (std::ostream &s, const config_t &c) {
  return c.put(s);
}

struct gid_t {
  gid_t(union ibv_gid gid) : gid(gid) {
  }

  std::ostream &put(std::ostream &s) const;

  union ibv_gid gid;
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
  union ibv_gid gid;    /* gid */
} __attribute__ ((packed));


struct rdma_peer {
  struct ibv_qp *qp;                  /* QP handle */
  struct ibv_port_attr port_attr;     /* IB port attributes */
  struct cm_con_data_t props;         /* values to connect to remote side */
  cm_con_data_t con_data;
};

template <typename ValueType>
struct rdma_area {
  rdma_area() : mr_(NULL), area_(NULL) {
  }

  ~rdma_area() {
    if (mr_ != NULL) {
      if (ibv_dereg_mr(mr_) != 0) {
        std::cerr << "failed to deregister MR" << std::endl;
      }
    }
    delete[] area_;
  }

  struct ibv_mr *mr_;                  /* MR handle for buf */
  ValueType *area_;
};


/* structure of system resources */
template <typename ValueType>
struct resources {

/* structure of system resources */
  resources()
      : ib_ctx(NULL), pd(NULL), cq(NULL) {
  }

  ~resources() {
    for (auto & p : peer_) {
      if (p.qp != NULL) {
        if (ibv_destroy_qp(p.qp) != 0) {
          std::cerr << "failed to destroy QP" << std::endl;
        }
      }
    }
    if (cache_.mr_ != NULL) {
      if (ibv_dereg_mr(cache_.mr_) != 0) {
        std::cerr << "failed to deregister Cache MR" << std::endl;
      }
    }
    delete[] value_.area_;
    if (cq != NULL) {
      if (ibv_destroy_cq (cq) != 0) {
        std::cerr << "failed to destroy CQ" << std::endl;
      }
    }
    if (pd != NULL) {
      if (ibv_dealloc_pd (pd) != 0)
      {
        std::cerr << "failed to deallocate PD" << std::endl;
      }
    }
    if (ib_ctx != NULL) {
      if (ibv_close_device (ib_ctx) != 0) {
        std::cerr << "failed to close device context" << std::endl;
      }
    }
  }

  void Init(config_t *config, ::size_t num_servers, ::size_t value_size,
            ::size_t total_values, ::size_t max_capacity) {
    struct ibv_device **dev_list;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_device *ib_dev = NULL;
    int mr_flags = 0;
    int num_devices;

    /* get device names in the system */
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
      throw RDMAException("failed to get IB devices list");
    }
    /* if there isn't any IB device in host */
    if (num_devices == 0) {
      throw RDMAException("found no IB devices");
    }
    std::cerr << "RDMA: found " << num_devices << " device(s)" << std::endl;

    /* search for the specific device we want to work with */
    for (auto i = 0; i < num_devices; i++) {
      if (config->dev_name == "") {
        config->dev_name = std::string(ibv_get_device_name(dev_list[i]));
        std::cerr << "device not specified, using first one found: " <<
          config->dev_name << std::endl;
      }
      if (ibv_get_device_name(dev_list[i]) == config->dev_name) {
        ib_dev = dev_list[i];
        break;
      }
    }
    /* if the device wasn't found in host */
    if (ib_dev == NULL) {
      throw RDMAException("IB device " + config->dev_name + " not found");
    }
    /* get device handle */

    ib_ctx = ibv_open_device(ib_dev);
    if (ib_ctx == NULL) {
      throw RDMAException("failed to open device " + config->dev_name);
    }

    /* We are now done with device list, free it */
    ibv_free_device_list(dev_list);
    dev_list = NULL;
    ib_dev = NULL;

    /* query port properties */
    for (::size_t i = 0; i < num_servers; i++) {
      if (ibv_query_port (ib_ctx, config->ib_port, &peer_[i].port_attr)) {
        throw RDMAException("ibv_query_port on port " +
                            std::to_string(config->ib_port) + " failed");
      }
    }

    /* allocate Protection Domain */
    pd = ibv_alloc_pd(ib_ctx);
    if (pd == NULL) {
      throw RDMAException("ibv_alloc_pd failed");
    }

    cq = ibv_create_cq (ib_ctx, num_servers, NULL, NULL, 0);
    if (cq == NULL) {
      throw RDMAException("failed to create CQ with " +
                          std::to_string(num_servers) + "entries");
    }

    /* allocate the memory buffer that will hold the data */
    ::size_t my_values = (total_values + num_servers - 1) / num_servers;
    value_.area_ = new ValueType[value_size * my_values];
    if (value_.area_ == NULL) {
      throw RDMAException("cannot allocate value area");
    }
    /* register the memory buffers */
    mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    value_.mr_ = ibv_reg_mr (pd, value_.area_,
                            value_size * my_values * sizeof(ValueType),
                            mr_flags);
    if (value_.mr_ == NULL) {
      throw RDMAException("ibv_reg_mr failed with mr_flags=0x" +
                          std::to_string(mr_flags));
    }
    std::cout << "MR was registered with addr=" << (void *)value_.area_ <<
      std::hex << ", lkey=0x" << value_.mr_->lkey <<
      ", rkey=0x" << value_.mr_->rkey <<
      ", flags=0x" << mr_flags << std::endl << std::dec;

    /* allocate the memory buffer that will hold the data */
    cache_.area_ = new ValueType[value_size * max_capacity];
    if (cache_.area_ == NULL) {
      throw RDMAException("cannot allocate cache area");
    }
  std::cerr << "FIXME parameters to MR call" << std::endl;
    /* register the memory buffer */
    mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE;
    cache_.mr_ = ibv_reg_mr (pd, cache_.area_,
                            value_size * max_capacity * sizeof(ValueType),
                            mr_flags);
    if (cache_.mr_ == NULL) {
      throw RDMAException("ibv_reg_mr failed with mr_flags=0x" +
                          std::to_string(mr_flags));
    }
    std::cout << "MR was registered with addr=" << (void *)cache_.area_ <<
      std::hex << ", lkey=0x" << cache_.mr_->lkey <<
      ", rkey=0x" << cache_.mr_->rkey <<
      ", flags=0x" << mr_flags << std::endl << std::dec;

    peer_.resize(num_servers);
    /* create the Queue Pair */
    memset (&qp_init_attr, 0, sizeof qp_init_attr);
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 1;
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.cap.max_send_wr = 1;
    qp_init_attr.cap.max_recv_wr = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    for (::size_t i = 0; i < num_servers; ++i) {
      peer_[i].qp = ibv_create_qp (pd, &qp_init_attr);
      if (peer_[i].qp == NULL) {
        throw RDMAException("failed to create QP");
      }
      std::cout << "QP[" << i << "] was created, QP number=0x" <<
        std::hex << peer_[i].qp->qp_num << std::endl << std::dec;
    }
  }

 private:
  struct ibv_device_attr device_attr; /* Device attributes */
  struct ibv_context *ib_ctx;         /* device handle */
  struct ibv_pd *pd;                  /* PD handle */
  struct ibv_cq *cq;                  /* CQ handle */
  rdma_area<ValueType> value_;
  rdma_area<ValueType> cache_;
  std::vector<rdma_peer> peer_;       /* state of connection to peer_[i] */
  /* memory buffer pointers, used for RDMA and send ops */

  friend class DKVStoreRDMA;
};


struct q_item_t {
  q_item_t() {
    wr.sg_list = &sg_list;
  }

  struct ibv_send_wr    wr;
  struct ibv_sge        sg_list;
  q_item_t             *next;
};


/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  virtual ~DKVStoreRDMA();

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_capacity,
                    const std::vector<std::string> &args);

  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
							 const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  virtual void FlushKVRecords(const std::vector<KeyType> &key);

  /**
   * Purge the cache area
   */
  virtual void PurgeKVRecords();

 private:
  int32_t HostOf(KeyType key);
  uint64_t OffsetOf(KeyType key);
  void ExchangePorts();
  void InitRDMA();
  void modify_qp_to_init(struct ibv_qp *qp);
  void modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
                        const gid_t &dgid);
  void modify_qp_to_rts(struct ibv_qp *qp);
  void post_receive(struct rdma_area<ValueType> *res, struct ibv_qp *qp);
  void connect_qp(resources<ValueType> *res);
  void check_ibv_status(int r, const struct ibv_send_wr *bad);
  void finish_completion_queue(::size_t n);

  ::size_t num_servers_;
  ::size_t my_rank_;
  
  config_t config;
  resources<ValueType> res;

  std::vector<ibv_send_wr> send_wr_;
  std::vector<ibv_sge> send_sge_;
  std::vector<ibv_send_wr> recv_wr_;
  std::vector<ibv_sge> recv_sge_;
  ::size_t current_recv_req_;
  ::size_t current_send_req_;
  std::vector<ibv_send_wr *> q_recv_front_;
  std::vector<ibv_send_wr *> q_send_front_;
};

}   // namespace DKVRDMA
}   // namespace DKV

#endif  // ndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
