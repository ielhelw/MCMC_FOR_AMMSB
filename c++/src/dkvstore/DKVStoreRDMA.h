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

#include "dkvstore/config.h"

#ifndef MCMC_ENABLE_RDMA
#error "This file should not be included if the project is not setup to support RDMA"
#endif

#include "DKVStore.h"

#include "mcmc/timer.h"

#include "qperf-rdma.h"

#include "OOBNetwork.h"


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

  void Init(const DEVICE *device, Buffer<ValueType> *buffer,
            ::size_t n_elements) {
    Init(device, n_elements);
    buffer->Init(area_, n_elements);
  }

  void Init(const DEVICE *device, std::vector<Buffer<ValueType> > *buffer,
            ::size_t n_elements_per_buffer) {
    Init(device, buffer->size() * n_elements_per_buffer);
    ValueType *area = area_;
    for (auto & b : *buffer) {
      b.Init(area, n_elements_per_buffer);
      area += n_elements_per_buffer;
    }
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

class DKVStoreRDMAOptions : public DKVStoreOptions {
 public:
  DKVStoreRDMAOptions();

  void Parse(const std::vector<std::string> &args) override;
  boost::program_options::options_description* GetMutable() override { return &desc_; }
  
  inline const std::string& dev_name() const { return dev_name_; }
  inline int ib_port() const { return ib_port_; }
  inline int mtu() const { return mtu_; }
  inline ::size_t post_send_chunk() const { return post_send_chunk_; }
  inline ::size_t batch_size() const { return batch_size_; }
  inline bool force_include_master() const { return force_include_master_; }
  inline const std::string& oob_server() const { return oob_server_; }
  inline uint32_t oob_port() const { return oob_port_; }
  inline ::size_t oob_num_servers() const { return oob_num_servers_; }
  
  inline void set_batch_size(::size_t val) { batch_size_ = val; }
  inline ::size_t* mutable_oob_num_servers() { return &oob_num_servers_; }

 private:
  std::string dev_name_;
  int ib_port_;
  int mtu_;
  ::size_t post_send_chunk_;
  ::size_t batch_size_;
  bool force_include_master_;
  std::string oob_server_;
  uint32_t oob_port_;
  ::size_t oob_num_servers_;
  boost::program_options::options_description desc_;

  friend std::ostream& operator<<(std::ostream& out, const DKVStoreRDMAOptions& opts);
};

inline std::ostream& operator<<(std::ostream& out, const DKVStoreRDMAOptions& opts) {
  out << opts.desc_;
  return out;
}

/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  DKVStoreRDMA(const std::vector<std::string> &args);

  virtual ~DKVStoreRDMA();

 private:
  static double GBs_from_time(
      const std::chrono::high_resolution_clock::duration &dt, int64_t bytes);

  static double GBs_from_time(const double &dt, int64_t bytes);

  static double GBs_from_timer(const Timer &timer, int64_t bytes);

 public:
  VIRTUAL void Init(::size_t value_size, ::size_t total_values,
                    ::size_t num_cache_buffers, ::size_t cache_buffer_capacity,
                    ::size_t max_write_capacity);

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


  VIRTUAL void ReadKVRecords(::size_t buffer, std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key);

  VIRTUAL void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  VIRTUAL void FlushKVRecords();
  VIRTUAL void PurgeKVRecords(::size_t buffer);

  VIRTUAL void barrier();

 private:
  int32_t HostOf(DKVStoreRDMA::KeyType key);

  uint64_t OffsetOf(DKVStoreRDMA::KeyType key);

  ::size_t PollForCookies(::size_t current, ::size_t at_least, BatchTimer &timer);

  void post_batches(const std::vector<std::vector<PostDescriptor<ValueType> > > &post_descriptor,
                    const std::vector< ::size_t> &posts,
                    uint32_t local_key,
                    enum ibv_wr_opcode opcode,
                    BatchTimer &timer);

  DKVStoreRDMAOptions options_;

  DEVICE res_;
  std::vector<rdma_peer> peer_;

  std::vector< ::ibv_wc> wc_;

  std::vector<std::vector<PostDescriptor<ValueType> > > post_descriptor_;
  std::vector< ::size_t> posts_;

  bool include_master_;	// if unset, the KV area is distributed over all nodes
                        // except the master. Watch out for the case #hosts == 1

  /* memory buffer pointers, used for RDMA and send ops */
  rdma_area<ValueType> value_;
  rdma_area<ValueType> cache_;
  rdma_area<ValueType> write_;

  ::size_t oob_rank_;
  OOBNetwork<cm_con_data_t> oob_network_;

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
