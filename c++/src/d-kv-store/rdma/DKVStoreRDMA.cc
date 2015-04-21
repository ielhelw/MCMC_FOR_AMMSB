/*
 * Copyright notice
 */

#include "d-kv-store/rdma/DKVStoreRDMA.h"

#include <stdlib.h>
#include <errno.h>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

#include <infiniband/verbs.h>

#include <iostream>
#include <iomanip>

#include <exception>

#include <mpi.h>

namespace DKV {
namespace DKVRDMA {

config_t::config_t() : dev_name(""), ib_port(1), gid_idx(-1) {
}

std::ostream &config_t::put(std::ostream &s) const {
  s << " ------------------------------------------------" << std::endl;
  s << " Device name : " << dev_name << std::endl;
  s << " IB port : " << ib_port << std::endl;
  if (gid_idx >= 0)
    s << " GID index : " << gid_idx << std::endl;
  s << " ------------------------------------------------" << std::endl <<
    std::endl;
  return s;
}

std::ostream &gid_t::put(std::ostream &s) const {
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


static void mpi_error_test(int r, const std::string &message) {
  if (r != MPI_SUCCESS) {
    throw NetworkException("MPI error " + r + message);
  }
}


void DKVStoreRDMA::Init(::size_t value_size, ::size_t total_values,
                        ::size_t max_capacity,
                        const std::vector<std::string> &args) {
  namespace po = ::boost::program_options;

  value_size_ = value_size;
  total_values_ = total_values;
  max_capacity_ = max_capacity;

std::cerr << "FIXME MPI_Init reentrant" << std::endl;
  int r;
  int n;
  r = MPI_Comm_size(MPI_COMM_WORLD, &n);
  mpi_error_test(r, "MPI_Comm_size() fails");
  num_servers_ = n;
  r = MPI_Comm_rank(MPI_COMM_WORLD, &n);
  mpi_error_test(r, "MPI_Comm_rank() fails");
  my_rank_ = n;

  po::options_description desc("RDMA options");
  desc.add_options()
    // ("rdma.oob-port", po::value(&config.tcp_port)->default_value(0), "RDMA out-of-band TCP port")
    ("rdma:dev",  po::value(&config.dev_name), "RDMA device name")
    ("rdma:port", po::value(&config.ib_port),  "RDMA device port")
    ("rdma:gid",  po::value(&config.gid_idx),  "RDMA GID index")
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

  /* print the used parameters for info*/
  std::cout << config;
  /* init all of the resources, so cleanup will be easy */
  /* create resources before using them */
  res.Init(&config, num_servers_, value_size_, total_values_, max_capacity_);

  /* connect the QPs */
  connect_qp (&res);
}

void DKVStoreRDMA::ReadKVRecords(std::vector<ValueType *> &cache,
                                 const std::vector<KeyType> &key,
                                 RW_MODE::RWMode rw_mode) {
}

void DKVStoreRDMA::FlushKVRecords(const std::vector<KeyType> &key) {
}

void DKVStoreRDMA::WriteKVRecords(const std::vector<KeyType> &key,
                                  const std::vector<const ValueType *> &value) {
}

void DKVStoreRDMA::PurgeKVRecords() {
}

int DKVStoreRDMA::HostOf(DKVStoreRDMA::KeyType key) {
  return key % num_servers_;
}

void DKVStoreRDMA::ExchangePorts() {
}

void DKVStoreRDMA::InitRDMA() {
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
 * RDMAException(ibv_modify_qp) failure
 *
 * Description
 * Transition a QP from the RESET to INIT state
 ******************************************************************************/
void DKVStoreRDMA::modify_qp_to_init (struct ibv_qp *qp) {
  struct ibv_qp_attr attr;
  int flags;
  memset (&attr, 0, sizeof (attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = config.ib_port;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
    IBV_ACCESS_REMOTE_WRITE;
  flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
    IBV_QP_ACCESS_FLAGS;
  int rc = ibv_modify_qp (qp, &attr, flags);
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
 * RDMAException(ibv_modify_qp) failure
 *
 * Description
 * Transition a QP from the INIT to RTR state, using the specified QP number
 ******************************************************************************/
void DKVStoreRDMA::modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
                      const gid_t &dgid) {
  struct ibv_qp_attr attr;
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
  int rc = ibv_modify_qp (qp, &attr, flags);
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
void DKVStoreRDMA::modify_qp_to_rts (struct ibv_qp *qp)
{
  struct ibv_qp_attr attr;
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
  int  rc = ibv_modify_qp (qp, &attr, flags);
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
void DKVStoreRDMA::post_receive(struct rdma_area<ValueType> *res, struct ibv_qp *qp) {
  struct ibv_recv_wr rr;
  struct ibv_sge sge;
  struct ibv_recv_wr *bad_wr;
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
  rc = ibv_post_recv (qp, &rr, &bad_wr);
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
  union ibv_gid my_gid;
  if (config.gid_idx >= 0) {
    rc = ibv_query_gid (res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
    if (rc != 0) {
      throw RDMAException("could not get gid for port " +
                          std::to_string(config.ib_port) + ", index " +
                          std::to_string(config.gid_idx));
    }
  } else {
    memset (&my_gid, 0, sizeof my_gid);
  }

  /* exchange using o-o-b network info required to connect QPs */
std::cerr << "FIXME: no Bcast: the QP status is personalized" << std::endl;
  std::vector<cm_con_data_t> my_con_data(num_servers_);
  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME: What about me?" << std::endl;
    }
    cm_con_data_t *con = &my_con_data[i];
    con->value  = (uint64_t) res->value_.area_;
    con->cache  = (uint64_t) res->cache_.area_;
    con->value_rkey   = res->value_.mr_->rkey;
    con->cache_rkey   = res->cache_.mr_->rkey;
    con->qp_num = res->peer_[i].qp->qp_num;
    con->lid    = res->peer_[i].port_attr.lid;
    con->gid    = my_gid;
    fprintf (stdout, "\nLocal LID[%zd] = 0x%x\n", i, res->peer_[i].port_attr.lid);
  }

  std::vector<cm_con_data_t> remote_con_data(num_servers_);
  int r;
  r = MPI_Alltoall(my_con_data.data(), sizeof my_con_data[0], MPI_BYTE,
                   remote_con_data.data(), sizeof remote_con_data[0], MPI_BYTE,
                   MPI_COMM_WORLD);
  mpi_error_test(r, "MPI conn alltoall");

  for (::size_t i = 0; i < num_servers_; ++i) {
    if (i == my_rank_) {
      std::cerr << "FIXME: What about me?" << std::endl;
    }
    res->peer_[i].con_data = remote_con_data[i];
    /* save the remote side attributes, we will need it for the post SR */
    res->peer_[i].props = remote_con_data[i];
    std::cout << std::hex;
    std::cout << "Remote value address = 0x" << remote_con_data[i].value << std::endl;
    std::cout << "Remote value rkey = 0x" << remote_con_data[i].value_rkey << std::endl;
    std::cout << "Remote cache address = 0x" << remote_con_data[i].cache << std::endl;
    std::cout << "Remote cache rkey = 0x" << remote_con_data[i].cache_rkey << std::endl;
    std::cout << "Remote QP number = 0x" << remote_con_data[i].qp_num << std::endl;
    std::cout << "Remote LID = 0x" << remote_con_data[i].lid << std::endl;
    if (config.gid_idx >= 0) {
      std::cout << gid_t(res->peer_[i].con_data.gid) << std::endl;
    }
    std::cout << std::dec;
  }

  /* modify the QP to init */
  for (::size_t i = 0; i < my_rank_; ++i) {
    if (i != my_rank_) {
      modify_qp_to_init(res->peer_[i].qp);
    } else {
      std::cerr << "FIXME: What about me?" << std::endl;
    }
  }

  // FIXME: What should we do about this? Do we _ever_ receive?
  /* let the client post RR to be prepared for incoming messages */
  for (::size_t i = 0; i < my_rank_; ++i) {
    if (i != my_rank_) {
      post_receive(&res->value_, res->peer_[i].qp);
      post_receive(&res->cache_, res->peer_[i].qp);
    } else {
      std::cerr << "FIXME: What about me?" << std::endl;
    }
  }
  fprintf (stderr, "Posted receive to QP\n");

  std::cerr << "FIXME: do we need a barrier here?" << std::endl;

  /* modify the QP to RTR */
  for (::size_t i = 0; i < my_rank_; ++i) {
    if (i != my_rank_) {
      modify_qp_to_rtr(res->peer_[i].qp,
                       res->peer_[i].con_data.qp_num,
                       res->peer_[i].con_data.lid,
                       res->peer_[i].con_data.gid);
    } else {
      std::cerr << "FIXME: What about me?" << std::endl;
    }
  }
  fprintf (stderr, "Modified QP state to RTR\n");

  std::cerr << "FIXME: do we need a barrier here?" << std::endl;

  for (::size_t i = 0; i < my_rank_; ++i) {
    modify_qp_to_rts(res->peer_[i].qp);
  }
  fprintf (stdout, "QP state was change to RTS\n");

  /* sync to make sure that both sides are in states that they can connect to prevent packet loose */
  r = MPI_Barrier(MPI_COMM_WORLD);
  mpi_error_test(r, "barrier after qp exchange");
}

#ifdef COPIED_EVERYTHING
/******************************************************************************
 * Function: main
 *
 * Input
 * argc number of items in argv
 * argv command line parameters
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, 1 on failure
 *
 * Description
 * Main program code
 ******************************************************************************/
int main (const std::vector<std::string> &args) {
  /* let the server post the sr */
  /*  if (!config.server_name)
      if (post_send (&res, IBV_WR_SEND))
      {
      fprintf (stderr, "failed to post sr\n");
      goto main_exit;
      }*/
  /* in both sides we expect to get a completion */
  /*  if (poll_completion (&res))
      {
      fprintf (stderr, "poll completion failed\n");
      goto main_exit;
      }*/
  /* after polling the completion we have the message in the client buffer too */
  if (config.server_name)
    fprintf (stdout, "Message is: '%s'\n", res.buf);
  else
  {
    /* setup server buffer with read message */
    strcpy (res.buf, RDMAMSGR);
  }
  /* Sync so we are sure server side has data ready before client tries to read it */
  if (sock_sync_data (res.sock, 1, "R", &temp_char))	/* just send a dummy char back and forth */
  {
    fprintf (stderr, "sync error before RDMA ops\n");
    rc = 1;
    goto main_exit;
  }
  /* Now the client performs an RDMA read and then write on server.
     Note that the server has no idea these events have occured */
  if (config.server_name)
  {
    /* First we read contens of server's buffer */
    if (post_send (&res, IBV_WR_RDMA_READ))
    {
      fprintf (stderr, "failed to post SR 2\n");
      rc = 1;
      goto main_exit;
    }
    if (poll_completion (&res))
    {
      fprintf (stderr, "poll completion failed 2\n");
      rc = 1;
      goto main_exit;
    }
    fprintf (stdout, "Contents of server's buffer: '%s'\n", res.buf);
    /* Now we replace what's in the server's buffer */
    strcpy (res.buf, RDMAMSGW);
    fprintf (stdout, "Now replacing it with: '%s'\n", res.buf);
    if (post_send (&res, IBV_WR_RDMA_WRITE))
    {
      fprintf (stderr, "failed to post SR 3\n");
      rc = 1;
      goto main_exit;
    }
    if (poll_completion (&res))
    {
      fprintf (stderr, "poll completion failed 3\n");
      rc = 1;
      goto main_exit;
    }
  }
  /* Sync so server will know that client is done mucking with its memory */
  if (sock_sync_data (res.sock, 1, "W", &temp_char))	/* just send a dummy char back and forth */
  {
    fprintf (stderr, "sync error after RDMA ops\n");
    rc = 1;
    goto main_exit;
  }
  if (!config.server_name)
    fprintf (stdout, "Contents of server buffer: '%s'\n", res.buf);
  rc = 0;
main_exit:
  if (resources_destroy (&res))
  {
    fprintf (stderr, "failed to destroy resources\n");
    rc = 1;
  }
  if (config.dev_name)
    free ((char *) config.dev_name);
  fprintf (stdout, "\ntest result is %d\n", rc);
  return rc;
}
#endif

}   // namespace DKVRDMA
}   // namespace KDV
