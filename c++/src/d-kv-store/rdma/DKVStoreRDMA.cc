/*
 * Copyright notice
 */
#ifdef ENABLE_RDMA
#include <d-kv-store/rdma/DKVStoreRDMA.h>
#endif

#include <chrono>
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop
#include <boost/lexical_cast.hpp>

#include <mcmc/random.h>
#include <mcmc/options.h>

#include <d-kv-store/file/DKVStoreFile.h>
#ifdef ENABLE_RAMCLOUD
#include <d-kv-store/ramcloud/DKVStoreRamCloud.h>
#endif
#ifdef ENABLE_RDMA
#include <infiniband/verbs.h>
#endif

#ifndef OK_TO_DEFINE_IN_CC_FILE
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

#include <iostream>
#include <iomanip>

#include <exception>

// #define USE_MPI
#if defined ENABLE_NETWORKING
#  ifdef USE_MPI
#    include <mpi.h>
#  else
#    include <mr/net/netio.h>
#  endif
#endif
#endif


namespace DKV {
namespace DKVRDMA {

struct ibv_device **global_dev_list = NULL;

#ifdef OK_TO_DEFINE_IN_CC_FILE
#ifdef USE_MPI

static void mpi_error_test(int r, const std::string &message) {
  if (r != MPI_SUCCESS) {
    throw NetworkException("MPI error " + r + message);
  }
}

void DKVStoreRDMA::init_networking() {
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
#endif


#ifndef DISABLE_NETWORKING
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
#endif


void DKVStoreRDMA::alltoall(const void *sendbuf, ::size_t send_item_size,
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


void DKVStoreRDMA::barrier() {
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



void DKVStoreRDMA::Info() const {
  CHECK_DEV_LIST();
}

}   // namespace DKV {
}   // namespace DKVRDMA {
