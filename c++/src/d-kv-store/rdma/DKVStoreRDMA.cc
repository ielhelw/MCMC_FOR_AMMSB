/*
 * Copyright notice
 */

#include "d-kv-store/rdma/DKVStoreRDMA.h"

#include <stdlib.h>
#include <errno.h>

#include <exception>

#include <das_inet_sync.h>

namespace DKV {
namespace DKVRDMA {

void DKVStoreRDMA::Init(::size_t value_size, ::size_t total_values,
                        const std::vector<std::string> &args) {
  value_size_ = value_size;
  total_values_ = total_values;
}

void DKVStoreRDMA::ReadKVRecords(
    const std::vector<DKVStoreRDMA::KeyType> &keys,
    const std::vector<DKVStoreRDMA::ValueType *> &cache) {
}

void DKVStoreRDMA::WriteKVRecords(
    const std::vector<DKVStoreRDMA::KeyType> &key,
    const std::vector<const DKVStoreRDMA::ValueType *> &cached) {
}

int DKVStoreRDMA::HostOf(DKVStoreRDMA::KeyType key) {
  return -1;
}

#ifdef NOT_YET
void DKVStoreRDMA::ExchangePorts() {
  const char *rank_env = getenv("PRUN_CPU_RANK");
  if (rank_env == NULL) {
    throw std::exception("Environment variable PRUN_CPU_RANK not defined");
  }
  std::ostringstream o(rank_env);
  o >> rank_;
  const char *hostlist_env = getenv("PRUN_HOSTNAMES");
  if (hostlist_env == NULL) {
    throw std::exception("Environment variable PRUN_HOSTNAMES not defined");
  }
  string hosts(hostlist_env);
  boost::trim(hosts);
  std::vector<std::string> host_vector;
  boost::split(host_vector, hosts, boost::is_any_of(" "),
               boost::token_compression);
  size_ = host_vector.size();
  if (true) {
    std::cerr << "I am " << rank_ << " of " << size_ <<
      " host " << host_vector[rank_] << std::endl;
  }

  int argc = 0;
  char *argv[] = { NULL };
  if (das_inet_sync_init(&argc, argv) == -1) {
    throw std::exception("Cannot DAS Inet Sync Init " + strerror(errno));
  }
  if (das_inet_sync_send()) {
  }
}

void DKVStoreRDMA::InitRDMA() {
  value_area_ = new ValueType[value_size_ * total_values_];
  if (value_area_ == NULL) {
    throw std::exception("Cannot allocate value_area_");
  }

  rdma_id_ = std::vector<rdma_cmd_id>(size_);
  struct ibv_qp_init_attr attr;
  attr.cap.max_send_wr = attr.cap.max_recv_wr = 1;
  attr.cap.max_send_sge = attr.cap.max_recv_sge = 1;
  // FIXME: this is too large...
  attr.cap.max_inline_data = value_size_ * total_values_ * sizeof value_area_[0];
  attr.sq_sig_all = 1;

  for (::size_t i = 0; i < size_; i++) {
    if (i != rank_) {
      attr.qp_context = &rmda_id_[i];
      if (rdma_create_ep(&rdma_ep_[i], )) {
      }
      if (rdma_getaddrinfo(host_vector[rank_], NULL, &hints, &res) == -1) {
        throw std::exception("Cannot rdma_getaddrinfo: " + strerror(errno));
      }
    }
  }
  value_handle_ = rmda_reg_msgs(id, value_area_,
                                value_size_ * total_values_ *
                                   sizeof value_area_[0]);
  if (value_handle_ == 0) {
    throw std::exception("Cannot register value_area_ handle");
  }
}
#endif

}   // namespace DKVRDMA
}   // namespace KDV
