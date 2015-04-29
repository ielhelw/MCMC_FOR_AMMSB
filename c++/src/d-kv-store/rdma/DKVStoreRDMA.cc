/*
 * Copyright notice
 */

#include "d-kv-store/rdma/DKVStoreRDMA.h"

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

#ifdef OK_TO_DEFINE_IN_CC_FILE
DKVStoreRDMA::~DKVStoreRDMA() {
#ifndef DISABLE_NETWORKING
#  ifndef USE_MPI
	delete network_;
#  endif
#endif
}

void DKVStoreRDMA::Init(::size_t value_size, ::size_t total_values,
                        ::size_t max_capacity,
                        const std::vector<std::string> &args) {
  CHECK_DEV_LIST();
  namespace po = ::boost::program_options;

  value_size_ = value_size;
  total_values_ = total_values;
  max_capacity_ = max_capacity;
  CHECK_DEV_LIST();

  po::options_description desc("RDMA options");
  desc.add_options()
    // ("rdma.oob-port", po::value(&config.tcp_port)->default_value(0), "RDMA out-of-band TCP port")
    ("rdma:dev",  po::value(&config.dev_name), "RDMA device name")
    ("rdma:port", po::value(&config.ib_port),  "RDMA device port")
    ("rdma:gid",  po::value(&config.gid_idx),  "RDMA GID index")
    ;
  CHECK_DEV_LIST();

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);
  CHECK_DEV_LIST();

  /* print the used parameters for info*/
  std::cout << config;

  init_networking();
  CHECK_DEV_LIST();

  /* init all of the resources, so cleanup will be easy */
  /* create resources before using them */
  res.Init(&config, my_rank_, num_servers_,
           value_size_, total_values_, max_capacity_);

  /* connect the QPs */
  connect_qp (&res);

  q_recv_front_.resize(num_servers_);
  q_send_front_.resize(num_servers_);
  recv_wr_.resize(max_capacity_);
  recv_sge_.resize(max_capacity_);
}


void DKVStoreRDMA::check_ibv_status(int r, const struct ::ibv_send_wr *bad) {
  if (r != 0) {
    std::cerr << "IBV status " << r << " bad " << bad << std::endl;
  }
}


void DKVStoreRDMA::finish_completion_queue(::size_t n) {
#ifndef DISABLE_NETWORKING
  struct ::ibv_wc wc;
  int r;
  for (::size_t i = 0; i < n; i++) {
    do {
      r = ::ibv_poll_cq(res.cq, 1, &wc);
      if (r < 0) {
        std::cerr << "Oops, must handle wrong result cq poll" << std::endl;
        break;
      }
    } while (r == 0);
  }
#endif
}


void DKVStoreRDMA::ReadKVRecords(std::vector<ValueType *> &cache,
                                 const std::vector<KeyType> &key,
                                 RW_MODE::RWMode rw_mode) {
  if (rw_mode != RW_MODE::READ_ONLY) {
    std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
  }

#ifndef DISABLE_NETWORKING
  // Fill the linked lists of WR requests
  ValueType *target = cache_ + next_free_;
  q_recv_front_.assign(num_servers_, NULL);
  current_recv_req_ = 0;
  assert(recv_wr_.capacity() >= key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    ::size_t owner = HostOf(key[i]);
    if (owner == my_rank_) {
      // Read directly, without RDMA
      cache[i] = res.value_.area_ + OffsetOf(key[i]);

    } else {
      cache[i] = target;

      struct ::ibv_send_wr *wr = &recv_wr_[current_recv_req_];
      struct ::ibv_sge *sge = &recv_sge_[current_recv_req_];
      current_recv_req_++;

      wr->num_sge = 1;
      wr->sg_list = sge;
      sge->addr   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(target));
      sge->length = value_size_ * sizeof(ValueType);
      sge->lkey   = res.cache_.mr_->lkey;

      wr->opcode = IBV_WR_RDMA_READ;
      wr->send_flags = 0;
      wr->wr.rdma.remote_addr = res.peer_[i].props.value +
                                 OffsetOf(key[i]) * sizeof(ValueType);
      wr->wr.rdma.rkey = res.peer_[i].props.value_rkey;

      wr->next = q_recv_front_[owner];
      q_recv_front_[owner] = wr;

      target += value_size_;
    }
  }

  next_free_ += value_size_ * key.size();

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
      std::cerr << "post " << sent_items << " read requests to host " << i << std::endl;

      int r = ::ibv_post_send(res.peer_[i].qp, q_recv_front_[i], &bad);
      check_ibv_status(r, bad);

      finish_completion_queue(sent_items);
    }
  }

  std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
#endif
}

void DKVStoreRDMA::FlushKVRecords(const std::vector<KeyType> &key) {
  std::cerr << "Ooppssss.......... writeable records not yet implemented" << std::endl;
}

void DKVStoreRDMA::WriteKVRecords(const std::vector<KeyType> &key,
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
      ValueType *target = res.value_.area_ + OffsetOf(key[i]);
      memcpy(target, value[i], value_size_ * sizeof(ValueType));

    } else {
      struct ::ibv_send_wr *wr = &send_wr_[current_send_req_];
      struct ::ibv_sge *sge = &recv_sge_[current_send_req_];
      current_send_req_++;

      wr->num_sge = 1;
      wr->sg_list = sge;
      sge->addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value[i]));
      sge->length = value_size_ * sizeof(ValueType);
      sge->lkey = res.cache_.mr_->lkey;

      wr->opcode = IBV_WR_RDMA_WRITE;
      wr->send_flags = 0;
      wr->wr.rdma.remote_addr = res.peer_[i].props.value +
                                OffsetOf(key[i]) * sizeof(ValueType);
      wr->wr.rdma.rkey = res.peer_[i].props.value_rkey;

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
      std::cerr << "post " << sent_items << " write requests to host " << i << std::endl;

      int r = ::ibv_post_send(res.peer_[i].qp, q_send_front_[i], &bad);
      check_ibv_status(r, bad);

      finish_completion_queue(sent_items);
    }
  }

  std::cerr << "FIXME: should I GC the recv/send queue items? " << std::endl;
#endif
}

void DKVStoreRDMA::PurgeKVRecords() {
  CHECK_DEV_LIST();
  next_free_ = 0;
}

int DKVStoreRDMA::HostOf(DKVStoreRDMA::KeyType key) {
  return key % num_servers_;
}

uint64_t DKVStoreRDMA::OffsetOf(DKVStoreRDMA::KeyType key) {
  return key / num_servers_;
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
void DKVStoreRDMA::modify_qp_to_init (struct ::ibv_qp *qp) {
  struct ::ibv_qp_attr attr;
  int flags;
  memset (&attr, 0, sizeof (attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = config.ib_port;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
    IBV_ACCESS_REMOTE_WRITE;
  flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
    IBV_QP_ACCESS_FLAGS;
  int rc = ::ibv_modify_qp (qp, &attr, flags);
  if (rc != 0) {
    throw RDMAException("failed to modify QP state to INIT");
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
void DKVStoreRDMA::modify_qp_to_rtr(struct ::ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
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
  attr.ah_attr.port_num = config.ib_port;
  if (config.gid_idx >= 0) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    memcpy (&attr.ah_attr.grh.dgid, &dgid.gid, 16);
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.sgid_index = config.gid_idx;
    attr.ah_attr.grh.traffic_class = 0;
  }
  flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  int rc = ::ibv_modify_qp (qp, &attr, flags);
  if (rc != 0) {
    throw RDMAException("failed to modify QP state to RTR");
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
void DKVStoreRDMA::modify_qp_to_rts (struct ::ibv_qp *qp)
{
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
    throw RDMAException("failed to modify QP state to RTS");
  }
}

/******************************************************************************
 * Function: post_receive
 *
 * Input
 * res pointer to resources structure
 *
 * Output
 * none
 *
 * @throws RDMAException
 *
 * Description
 *
 ******************************************************************************/
void DKVStoreRDMA::post_receive(struct rdma_area<ValueType> *res, struct ::ibv_qp *qp) {
  struct ::ibv_recv_wr rr;
  struct ::ibv_sge sge;
  struct ::ibv_recv_wr *bad_wr = NULL;
  int rc;
  /* prepare the scatter/gather entry */
  memset (&sge, 0, sizeof (sge));
  sge.addr = (uintptr_t) res->area_;
#define MSG_SIZE 1024
  std::cerr << "FIXME: determine this SGE thingy" << std::endl;
  sge.length = MSG_SIZE;
  sge.lkey = res->mr_->lkey;
  /* prepare the receive work request */
  memset (&rr, 0, sizeof (rr));
  rr.next = NULL;
  rr.wr_id = 0;
  rr.sg_list = &sge;
  rr.num_sge = 1;
  /* post the Receive Request to the RQ */
  rc = ::ibv_post_recv (qp, &rr, &bad_wr);
  if (rc != 0) {
    throw RDMAException("failed to post RR");
  }
}

/******************************************************************************
 * Function: connect_qp
 *
 * Input
 * res pointer to resources structure
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, error code on failure
 *
 * Description
 * Connect the QP. Transition the server side to RTR, sender side to RTS
 ******************************************************************************/
void DKVStoreRDMA::connect_qp(struct resources<ValueType> *res) {
  int rc = 0;
  union ::ibv_gid my_gid;
  if (config.gid_idx >= 0) {
    rc = ::ibv_query_gid (res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
    if (rc != 0) {
      throw RDMAException("could not get gid for port " +
                          to_string(config.ib_port) + ", index " +
                          to_string(config.gid_idx));
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
      con->value  = (uint64_t) res->value_.area_;
      con->cache  = (uint64_t) res->cache_.area_;
      con->value_rkey   = res->value_.mr_->rkey;
if (0) {
      con->cache_rkey   = res->cache_.mr_->rkey;
} else {
  std::cerr << "DEBUG: skip cache region" << std::endl;
}
      con->qp_num = res->peer_[i].qp->qp_num;
      con->lid    = res->peer_[i].port_attr.lid;
      con->gid    = my_gid;
      fprintf (stdout, "\nLocal LID[%zd] = 0x%x\n", i, res->peer_[i].port_attr.lid);
    }
  }

  std::vector<cm_con_data_t> remote_con_data(num_servers_);
  alltoall(my_con_data.data(), sizeof my_con_data[0],
		   remote_con_data.data(), sizeof remote_con_data[0]);

  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      res->peer_[i].con_data = remote_con_data[i];
      /* save the remote side attributes, we will need it for the post SR */
      res->peer_[i].props = remote_con_data[i];

      std::cout << "Peer " << i << std::endl;
      std::cout << remote_con_data[i];
      if (config.gid_idx >= 0) {
        std::cout << "  gid " << gid_t(res->peer_[i].con_data.gid) << std::endl;
      }
      std::cout << std::dec;
    }
  }

  /* modify the QP to init */
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      modify_qp_to_init(res->peer_[i].qp);
    }
  }

  // FIXME: What should we do about this? Do we _ever_ receive?
  /* let the client post RR to be prepared for incoming messages */
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      post_receive(&res->value_, res->peer_[i].qp);
      post_receive(&res->cache_, res->peer_[i].qp);
    }
  }
  fprintf (stderr, "Posted receive to QP\n");

  std::cerr << "FIXME: do we need a barrier here?" << std::endl;

  /* modify the QP to RTR */
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      modify_qp_to_rtr(res->peer_[i].qp,
                       res->peer_[i].con_data.qp_num,
                       res->peer_[i].con_data.lid,
                       res->peer_[i].con_data.gid);
    }
  }
  fprintf (stderr, "Modified QP state to RTR\n");

  std::cerr << "FIXME: do we need a barrier here?" << std::endl;

  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME " << __LINE__ << ": What about me?" << std::endl;
    } else {
      modify_qp_to_rts(res->peer_[i].qp);
    }
  }
  fprintf (stderr, "QP state was change to RTS\n");

  /* sync to make sure that both sides are in states that they can connect to prevent packet loose */
  barrier();
}
#endif

}   // namespace DKV {
}   // namespace DKVRDMA {
