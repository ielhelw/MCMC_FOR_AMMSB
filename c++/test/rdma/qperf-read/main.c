#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <errno.h>

#include "pan_timer.h"
#include "mrg_GP_socket.h"

#include <d-kv-store/rdma/qperf-rdma.h>

#ifndef HOST_NAME_MAX
#  define HOST_NAME_MAX 256
#endif

typedef struct RDMA {
  DEVICE        device;
  REGION        value;
  REGION        cache;
  REGION        peer_value;
  REGION        peer_cache;
} rdma_t;


typedef struct CM_CON_DATA {
  uint64_t              value;
  uint64_t              cache;
  uint32_t              value_rkey;
  uint32_t              cache_rkey;
  uint32_t              qp_num;
  uint32_t              psn;
  uint16_t              lid;
  uint16_t              alt_lid;
} cm_con_data_t;


static int separate_cache = 1;


static int
cm_con_data_print(FILE *out, const cm_con_data_t *con)
{
  fprintf(out, "  value address = 0x%" PRIx64 "\n", con->value);
  fprintf(out, "  value rkey = 0x%" PRIx32 "\n",    con->value_rkey);
  fprintf(out, "  cache address = 0x%" PRIx64 "\n", con->cache);
  fprintf(out, "  cache rkey = 0x%" PRIx32 "\n",    con->cache_rkey);
  fprintf(out, "  QP number = 0x%" PRIx32 "\n",     con->qp_num);
  fprintf(out, "  PSN number = 0x%" PRIx32 "\n",    con->psn);
  fprintf(out, "  LID = 0x%" PRIx16 "\n",           con->lid);

  return 0;
}


static void
bail_out(const char *msg)
{
  fprintf(stderr, "Fail: %s %s error %s\n", msg, rd_get_error_message(), strerror(errno));
  exit(33);
}


static int
read_fully(int fd, void *to, size_t size)
{
  size_t        rd = 0;

  while (rd < size) {
    ssize_t r = read(fd, (char *)to + rd, size - rd);
    if (r == -1) {
     if (errno == EINTR) {
      // ignore
     } else {
       return -1;
     }
    } else {
      rd += r;
    }
  }

  return 0;
}


static int
write_fully(int fd, const void *to, size_t size)
{
  size_t        rd = 0;

  while (rd < size) {
    ssize_t r = write(fd, (const char *)to + rd, size - rd);
    if (r == -1) {
     if (errno == EINTR) {
      // ignore
     } else {
       return -1;
     }
    } else {
      rd += r;
    }
  }

  return 0;
}


static size_t
min(size_t a, size_t b)
{
  return (a < b) ? a : b;
}


static int
rdma_sync(int sock, int up_else_down, rdma_t *rdma, CONNECTION *me)
{
  cm_con_data_t con;
  con.value      = (uint64_t)(rdma->value.vaddr);
  con.cache      = (uint64_t)(rdma->cache.vaddr);
  con.value_rkey = rdma->value.mr->rkey;
  if (separate_cache) {
    con.cache_rkey = rdma->cache.mr->rkey;
  }
  con.qp_num     = me->local.qpn;
  con.psn        = me->local.psn;
  con.lid        = rdma->device.lnode.lid;
  con.alt_lid    = rdma->device.lnode.alt_lid;

  fprintf(stdout, "Me:\n");
  cm_con_data_print(stdout, &con);

  cm_con_data_t peer_con;

  if (up_else_down) {
    if (read_fully(sock, &peer_con, sizeof peer_con) == -1) {
      return -1;
    }
    if (write_fully(sock, &con, sizeof con) == -1) {
      return -1;
    }
  } else {
    if (write_fully(sock, &con, sizeof con) == -1) {
      return -1;
    }
    if (read_fully(sock, &peer_con, sizeof peer_con) == -1) {
      return -1;
    }
  }

  rdma->peer_value.vaddr = (void *)peer_con.value;
  rdma->peer_cache.vaddr = (void *)peer_con.cache;
  rdma->peer_value.key   = peer_con.value_rkey;
  rdma->peer_cache.key   = peer_con.cache_rkey;
  me->remote.qpn       = peer_con.qp_num;
  me->remote.psn       = peer_con.psn;
  me->rnode.lid        = peer_con.lid;
  me->rnode.alt_lid    = peer_con.alt_lid;

  fprintf(stdout, "Peer:\n");
  cm_con_data_print(stdout, &peer_con);

  return 0;
}


int main(int argc, char *argv[])
{
  size_t        K = 1024;
  size_t        N = 1024;
  size_t        neighb = 1024;
  size_t        q_items = 1024;
  size_t        iterations = 1024;
  rdma_t        rdma;
  CONNECTION    connection;
  const char   *server = NULL;
  int           sock;
  int           single_pointer = 0;
  int           is_client = 0;
  char          server_name[HOST_NAME_MAX + 8];

  das_time_init(&argc, argv);

  // fill Req
  Req.mtu_size = 2048;
  Req.id = "";
  Req.port = 1;
  Req.static_rate = "";
  Req.src_path_bits = 0;
  Req.sl = 0;
  Req.poll_mode = 1;
  Req.alt_port = 0;

  // parse options
  int option = 0;
  for (int i = 1; i < argc; i++) {
    if (0) {
    } else if (i < argc - 1 && strcmp(argv[i], "-K") == 0) {
      i++;
      if (sscanf(argv[i], "%zd", &K) != 1) {
        bail_out("K must be a size_t");
      }
    } else if (i < argc - 1 && strcmp(argv[i], "-N") == 0) {
      i++;
      if (sscanf(argv[i], "%zd", &N) != 1) {
        bail_out("N must be a size_t");
      }
    } else if (i < argc - 1 && strcmp(argv[i], "-n") == 0) {
      i++;
      if (sscanf(argv[i], "%zd", &neighb) != 1) {
        bail_out("n must be a size_t");
      }
    } else if (i < argc - 1 && strcmp(argv[i], "-x") == 0) {
      i++;
      if (sscanf(argv[i], "%zd", &iterations) != 1) {
        bail_out("x must be a size_t");
      }
    } else if (i < argc - 1 && strcmp(argv[i], "--chunk") == 0) {
      i++;
      if (sscanf(argv[i], "%zd", &q_items) != 1) {
        bail_out("chunk must be a size_t");
      }
    } else if (i < argc - 1 && strcmp(argv[i], "--mtu") == 0) {
      i++;
      if (sscanf(argv[i], "%d", &Req.mtu_size) != 1) {
        bail_out("mtu must be an int");
      }
    } else if (strcmp(argv[i], "-c") == 0) {
      is_client = 1;
    } else if (strcmp(argv[i], "-C") == 0) {
      separate_cache = 0;
    } else if (strcmp(argv[i], "-1") == 0) {
      single_pointer = 1;
    } else if (option == 0) {
      server = argv[i];
      option++;
    } else {
      fprintf(stderr, "Unknown option: \"%s\"\n", argv[i]);
      bail_out("Unknown option");
    }
  }

  fprintf(stdout, "K %zu (%zu bytes) N %zu n %zu x %zu\n",
          K, K * sizeof(double), N, neighb, iterations);
  fprintf(stdout, "IB: mtu %d id \"%s\" port %d rate \"%s\" poll %d alt-port %d\n",
          Req.mtu_size, Req.id, Req.port, Req.static_rate, Req.poll_mode, Req.alt_port);

  // init ib
  if (rd_open(&rdma.device, IBV_QPT_RC, q_items, 0) != 0) {
    bail_out("rd_open");
  }

  // create send/receive regions
  if (rd_mralloc(&rdma.value, &rdma.device, N * K * sizeof(double)) != 0) {
    bail_out("rd_mralloc(value)");
  }
  if (separate_cache) {
    if (rd_mralloc(&rdma.cache, &rdma.device, neighb * K * sizeof(double)) != 0) {
      bail_out("rd_mralloc(cache)");
    }
  }

  if (rd_create_qp(&rdma.device, &connection, rdma.device.ib.context, NULL) != 0) {
    bail_out("rd_create_qp");
  }
  if (rd_open_2(&rdma.device, &connection) != 0) {
    bail_out("rd_open_2");
  }

  if (is_client && server == NULL) {
    server = server_name;
    fprintf(stdout, "Server name/port: ");
    fflush(stdout);
    fgets(server_name, sizeof server_name, stdin);
  }

  // set up o-o-b network
  if (server == NULL) {
    int srv = mrg_socket_server(server_name, sizeof server_name);
    if (srv == -1) {
      bail_out("cannot create server socket");
    }
    fprintf(stdout, "Server socket: %s\n", server_name);
    fflush(stdout);
    sock = mrg_socket_accept(srv);
    if (sock == -1) {
      bail_out("cannot accept on server socket");
    }
  } else {
    sock = mrg_socket_client(server);
    if (sock == -1) {
      bail_out("cannot connect");
    }
  }

  rdma_sync(sock, (server == NULL), &rdma, &connection);

  fprintf(stdout, "Migrate QP to RTR, RTS\n");
  if (rd_prep(&rdma.device, &connection) != 0) {
    bail_out("rd_prep");
  }

  if (server != NULL) {
    const void **local_address = malloc(neighb * sizeof *local_address);
    if (local_address == NULL) {
      bail_out("malloc(local_address)");
    }
    const void **remote_address = malloc(neighb * sizeof *remote_address);
    if (local_address == NULL) {
      bail_out("malloc(remote_address)");
    }
    size_t *sizes = malloc(neighb * sizeof *sizes);
    if (local_address == NULL) {
      bail_out("malloc(sizes)");
    }
    struct ibv_wc *wc = malloc(q_items * sizeof *wc);
    if (wc == NULL) {
      bail_out("malloc(wc)");
    }

    for (size_t i = 0; i < neighb; i++) {
      if (separate_cache) {
        local_address[i] = rdma.cache.vaddr;
      } else {
        local_address[i] = rdma.value.vaddr;
      }
      remote_address[i] = rdma.peer_value.vaddr;
      sizes[i] = K * sizeof(double);
    }

    pan_timer_t t_read = { 0, };
    pan_timer_t t_post_read = { 0, };
    pan_timer_t t_finish_read = { 0, };

    pan_timer_start(&t_read);
    for (size_t x = 0; x < iterations; x++) {
      size_t at = 0;
      size_t seen = 0;
      size_t n = q_items;
      while (seen < neighb) {
        size_t p = min(n, neighb - at);
        if (p > 0) {
          pan_timer_start(&t_post_read);
          if (single_pointer) {
            if (rd_post_rdma_std_1(&connection,
                                   separate_cache ? rdma.cache.mr->lkey : rdma.value.mr->lkey,
                                   rdma.peer_value.key,
                                   IBV_WR_RDMA_READ,
                                   p,      
                                   local_address[0],
                                   remote_address[0],
                                   sizes[0]) != 0) { 
              bail_out("rd_post_rdma_std_1");
            }         
          } else {
            if (rd_post_rdma_std(&connection,
                                 separate_cache ? rdma.cache.mr->lkey : rdma.value.mr->lkey,
                                 rdma.peer_value.key,
                                 IBV_WR_RDMA_READ,
                                 p,      
                                 local_address + at,
                                 remote_address + at,
                                 sizes + at) != 0) { 
              bail_out("rd_post_rdma_std");
            }         
          }
          pan_timer_stop(&t_post_read);
          at += p;
        }         

        pan_timer_start(&t_finish_read);
        n = rd_poll(&rdma.device, wc, q_items);
        pan_timer_stop(&t_finish_read);
        for (size_t i = 0; i < n; i++) {
          int status = wc[i].status;
          if (status != IBV_WC_SUCCESS) {
            bail_out("rd_poll");
          }         
        }         
        seen += n;
      }         
    }
    pan_timer_stop(&t_read);

    free(local_address);
    free(remote_address);
    free(sizes);

    size_t bytes = iterations * K * neighb * sizeof(double);
    fprintf(stdout, "Total time %.06fs\n", das_time_t2d(&t_read.total));
#define GIGA ((double)(1LL << 30))
    double mb = (double)bytes / das_time_t2d(&t_read.total) / GIGA;
    fprintf(stdout, "Bytes %zd throughput %.03fGB/s\n", bytes, mb);
  }

  // Abuse this for a barrier
  rdma_sync(sock, (server == NULL), &rdma, &connection);

  if (rd_close_qp(&connection) != 0) {
    bail_out("rd_close_qp");
  }

  // create send/receive regions
  if (rd_mrfree(&rdma.value, &rdma.device) != 0) {
    bail_out("rd_mralloc(value)");
  }
  if (separate_cache) {
    if (rd_mrfree(&rdma.cache, &rdma.device) != 0) {
      bail_out("rd_mralloc(cache)");
    }
  }

  if (rd_close(&rdma.device) != 0) {
    bail_out("rd_close");
  }
  if (rd_close_2(&rdma.device) != 0) {
    bail_out("rd_close_2");
  }

  return 0;
}
