/*
 * Copyright
 */
#ifndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
#define APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__

// #define OK_TO_DEFINE_IN_CC_FILE
// #ifdef OK_TO_DEFINE_IN_CC_FILE
// #define VIRTUAL virtual
// #else
#define VIRTUAL
// #endif

#include <string.h>		// strerror
#include <errno.h>
#include <inttypes.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <chrono>

#ifdef ENABLE_RDMA
#include <infiniband/verbs.h>
#endif

#include "mcmc/timer.h"

#if 0 && defined ENABLE_DISTRIBUTED
#  define USE_MPI
#endif

#ifdef USE_MPI
#  include <mpi.h>
#endif

#include "dkvstore/DKVStore.h"

#include "dkvstore/qperf-rdma.h"


namespace DKV {
namespace DKVRDMA {

using ::mcmc::timer::Timer;


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
    buffer->Init(area_, n_elements);
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
  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  DKVStoreRDMA();

  virtual ~DKVStoreRDMA();

 private:
  static double GBs_from_time(
      const std::chrono::high_resolution_clock::duration &dt, int64_t bytes);

  static double GBs_from_time(const double &dt, int64_t bytes);

  static double GBs_from_timer(const Timer &timer, int64_t bytes);

 public:
  VIRTUAL void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args);

  virtual bool include_master() {
    return include_master_;
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
                             RW_MODE::RWMode rw_mode);

  VIRTUAL void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  VIRTUAL std::vector<ValueType *> GetWriteKVRecords(::size_t n);

  VIRTUAL void FlushKVRecords(const std::vector<KeyType> &key);

  /**
   * Purge the cache area
   */
  VIRTUAL void PurgeKVRecords();

 private:
  int32_t HostOf(DKVStoreRDMA::KeyType key);

  uint64_t OffsetOf(DKVStoreRDMA::KeyType key);

  ::size_t PollForCookies(::size_t current, ::size_t at_least, BatchTimer &timer);

  void post_batches(const std::vector<std::vector<PostDescriptor<ValueType>>> &post_descriptor,
                    const std::vector< ::size_t> &posts,
                    uint32_t local_key,
                    enum ibv_wr_opcode opcode,
                    BatchTimer &timer);

  void init_networking();
#ifdef USE_MPI
  static void mpi_error_test(int r, const std::string &message);
#else
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
  VIRTUAL void barrier();

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

  bool include_master_;	// if unset, the KV area is distributed over all nodes
  						// except the master. Watch out for the case #hosts == 1

  /* memory buffer pointers, used for RDMA and send ops */
  rdma_area<ValueType> value_;
  rdma_area<ValueType> cache_;
  rdma_area<ValueType> write_;

#ifdef USE_MPI
  bool mpi_initialized = false;
#else
  std::string oob_impl_;
  std::string oob_interface_;
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
