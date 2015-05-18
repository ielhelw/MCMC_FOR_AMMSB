/*
 * qperf - handle RDMA tests.
 *
 * Copyright (c) 2002-2009 Johann George.  All rights reserved.
 * Copyright (c) 2006-2009 QLogic Corporation.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "d-kv-store/rdma/qperf-rdma.h"

#define _GNU_SOURCE
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <netinet/in.h>
// #include <rdma/rdma_cma.h>
// #include <infiniband/verbs.h>
// #include "qperf.h"

// Stuff from qperf.h

#define SUCCESS0	0

/*
 *  * Error actions.
 *   */
#define BUG 1                           /* Internal error */
#define SYS 2                           /* System error */
#define RET 4                           /* Return, don't exit */


static char error_message[256];
static int	Debug = 1;

Options Req;

/*
 * Add a string to a buffer.
 */
static void
buf_app(char **pp, char *end, char *str)
{
    char *p = *pp;
    int n = strlen(str);
    int l = end - p;

    if (n > l)
	n = l;
    memcpy(p, str, n);
    *pp = p + n;
}

/*
 * End a buffer.
 */
static void
buf_end(char **pp, char *end)
{
    char *p = *pp;

    if (p == end) {
	char *s = " ...";
	int n = strlen(s);
	memcpy(--p-n, s, n);
    }
    *p = '\n';
    *pp = p;
}

const char *rd_get_error_message(void) {
    return error_message;
}


/*
 * Print out an error message.  actions contain a set of flags that determine
 * what needs to get done.  If BUG is set, it is an internal error.  If SYS is
 * set, a system error is printed.  If RET is set, we return rather than exit.
 */
static int
error(int actions, const char *fmt, ...)
{
    va_list alist;
    char *p = error_message;
    char *q = p + sizeof(error_message);

    if ((actions & BUG) != 0)
	buf_app(&p, q, "internal error: ");
    va_start(alist, fmt);
    p += vsnprintf(p, q-p, fmt, alist);
    va_end(alist);
    if ((actions & SYS) != 0 && errno) {
	buf_app(&p, q, ": ");
	buf_app(&p, q, strerror(errno));
    }
    buf_end(&p, q);
    fwrite(error_message, 1, p+1-error_message, stdout);
    if ((actions & RET) != 0)
	return 0;

#if 0
    if (RemoteFD >= 0) { 
	send(RemoteFD, "?", 1, MSG_OOB);
	(void) write(RemoteFD, error_message, p-error_message);
	shutdown(RemoteFD, SHUT_WR);
	timeout_set(ERROR_TIMEOUT, sig_alrm_die);
	while (read(RemoteFD, error_message, sizeof(error_message)) > 0)
	    ;       
    }
#endif
    return -1;
}

/*
 * Print out a debug message.
 */
void
debug(char *fmt, ...)
{
    va_list alist;

    if (!Debug)
	return;
    va_start(alist, fmt);
    vfprintf(stderr, fmt, alist);
    va_end(alist);
    fprintf(stderr, "\n");
    fflush(stderr);
}



#define cardof(m)	(sizeof (m) / sizeof (m)[0])

/*
 * List of errors we can get from a CQE.
 */
NAMES CQErrors[] ={
    { IBV_WC_SUCCESS,                   "Success"                       },
    { IBV_WC_LOC_LEN_ERR,               "Local length error"            },
    { IBV_WC_LOC_QP_OP_ERR,             "Local QP operation failure"    },
    { IBV_WC_LOC_EEC_OP_ERR,            "Local EEC operation failure"   },
    { IBV_WC_LOC_PROT_ERR,              "Local protection error"        },
    { IBV_WC_WR_FLUSH_ERR,              "WR flush failure"              },
    { IBV_WC_MW_BIND_ERR,               "Memory window bind failure"    },
    { IBV_WC_BAD_RESP_ERR,              "Bad response"                  },
    { IBV_WC_LOC_ACCESS_ERR,            "Local access failure"          },
    { IBV_WC_REM_INV_REQ_ERR,           "Remote invalid request"        },
    { IBV_WC_REM_ACCESS_ERR,            "Remote access failure"         },
    { IBV_WC_REM_OP_ERR,                "Remote operation failure"      },
    { IBV_WC_RETRY_EXC_ERR,             "Retries exceeded"              },
    { IBV_WC_RNR_RETRY_EXC_ERR,         "RNR retry exceeded"            },
    { IBV_WC_LOC_RDD_VIOL_ERR,          "Local RDD violation"           },
    { IBV_WC_REM_INV_RD_REQ_ERR,        "Remote invalid read request"   },
    { IBV_WC_REM_ABORT_ERR,             "Remote abort"                  },
    { IBV_WC_INV_EECN_ERR,              "Invalid EECN"                  },
    { IBV_WC_INV_EEC_STATE_ERR,         "Invalid EEC state"             },
    { IBV_WC_FATAL_ERR,                 "Fatal error"                   },
    { IBV_WC_RESP_TIMEOUT_ERR,          "Responder timeout"             },
    { IBV_WC_GENERAL_ERR,               "General error"                 },
};


/*
 * Opcodes.
 */
NAMES Opcodes[] ={
    { IBV_WR_ATOMIC_CMP_AND_SWP,        "compare and swap"              },
    { IBV_WR_ATOMIC_FETCH_AND_ADD,      "fetch and add"                 },
    { IBV_WR_RDMA_READ,                 "rdma read"                     },
    { IBV_WR_RDMA_WRITE,                "rdma write"                    },
    { IBV_WR_RDMA_WRITE_WITH_IMM,       "rdma write with immediate"     },
    { IBV_WR_SEND,                      "send"                          },
    { IBV_WR_SEND_WITH_IMM,             "send with immediate"           },
};


/*
 * Events from the Connection Manager.
 */
NAMES CMEvents[] ={
    { RDMA_CM_EVENT_ADDR_RESOLVED,      "Address resolved"              },
    { RDMA_CM_EVENT_ADDR_ERROR,         "Address error"                 },
    { RDMA_CM_EVENT_ROUTE_RESOLVED,     "Route resolved"                },
    { RDMA_CM_EVENT_ROUTE_ERROR,        "Route error"                   },
    { RDMA_CM_EVENT_CONNECT_REQUEST,    "Connect request"               },
    { RDMA_CM_EVENT_CONNECT_RESPONSE,   "Connect response"              },
    { RDMA_CM_EVENT_CONNECT_ERROR,      "Connect error"                 },
    { RDMA_CM_EVENT_UNREACHABLE,        "Event unreachable"             },
    { RDMA_CM_EVENT_REJECTED,           "Event rejected"                },
    { RDMA_CM_EVENT_ESTABLISHED,        "Event established"             },
    { RDMA_CM_EVENT_DISCONNECTED,       "Event disconnected"            },
    { RDMA_CM_EVENT_DEVICE_REMOVAL,     "Device removal"                },
    { RDMA_CM_EVENT_MULTICAST_JOIN,     "Multicast join"                },
    { RDMA_CM_EVENT_MULTICAST_ERROR,    "Multicast error"               },
};


/*
 * Opcodes.
 */
RATES Rates[] ={
    { "",       IBV_RATE_MAX        },
    { "max",    IBV_RATE_MAX        },
    { "1xSDR",  IBV_RATE_2_5_GBPS   },
    { "1xDDR",  IBV_RATE_5_GBPS     },
    { "1xQDR",  IBV_RATE_10_GBPS    },
    { "4xSDR",  IBV_RATE_10_GBPS    },
    { "4xDDR",  IBV_RATE_20_GBPS    },
    { "4xQDR",  IBV_RATE_40_GBPS    },
    { "8xSDR",  IBV_RATE_20_GBPS    },
    { "8xDDR",  IBV_RATE_40_GBPS    },
    { "8xQDR",  IBV_RATE_80_GBPS    },
    { "2.5",    IBV_RATE_2_5_GBPS   },
    { "5",      IBV_RATE_5_GBPS     },
    { "10",     IBV_RATE_10_GBPS    },
    { "20",     IBV_RATE_20_GBPS    },
    { "30",     IBV_RATE_30_GBPS    },
    { "40",     IBV_RATE_40_GBPS    },
    { "60",     IBV_RATE_60_GBPS    },
    { "80",     IBV_RATE_80_GBPS    },
    { "120",    IBV_RATE_120_GBPS   },
};

static int     ib_open(DEVICE *dev);
#ifdef UNUSED
static int     ib_create_qp(CONNECTION *con, DEVICE *dev, const NODE *peer);
#endif
static int     ib_prep(const DEVICE *dev, CONNECTION *conn);
static int     ib_close1(DEVICE *dev);
static int     ib_close2(DEVICE *dev);

/*
 * This routine is never called and is solely to avoid compiler warnings for
 * functions that are not currently being used.
 */
void
rdma_not_called(void)
{
}


/*
 * Measure RDMA bandwidth (client side).
 */
int
rd_client_rdma_bw(DEVICE *dev,
		  CONNECTION *con,
		  size_t n_req,
		  const void **local_addr,
		  const void **remote_addr,
		  const size_t *sizes)
{
#ifdef COMES_LATER
    rd_post_rdma_std(dev, opcode, NCQE);
    while (!Finished) {
	int i;
	struct ibv_wc wc[NCQE];
	int n = rd_poll(&dev, wc, cardof(wc));

        if (Finished)
            break;
        if (n > LStat.max_cqes)
            LStat.max_cqes = n;
        for (i = 0; i < n; ++i) {
            int status = wc[i].status;

            if (status == IBV_WC_SUCCESS) {
                if (opcode == IBV_WR_RDMA_READ) {
                    LStat.r.no_bytes += dev.msg_size;
                    LStat.r.no_msgs++;
                    LStat.rem_s.no_bytes += dev.msg_size;
                    LStat.rem_s.no_msgs++;
                    if (Req.access_recv)
                        touch_data(dev.buffer, dev.msg_size);
                }
            } else
                do_error(status, &LStat.s.no_errs);
        }
        rd_post_rdma_std(&dev, opcode, n);
    }
    stop_test_timer();
    exchange_results();
    rd_close(&dev);
    fprintf(stderr, "Max n %d\n", LStat.max_cqes);
#endif
    return 0;
}


/*
 * Open a RDMA device.
 */
int
rd_open(DEVICE *dev, int trans, int max_send_wr, int max_recv_wr)
{
#if 0
    /* Send request to client */
    if (is_client())
        client_send_request();
#endif

    /* Clear structure */
    memset(dev, 0, sizeof(*dev));

    /* Set transport type and maximum work request parameters */
#if 0
    dev->trans = trans;
#endif
    dev->max_send_wr = max_send_wr;
    dev->max_recv_wr = max_recv_wr;

    /* Open device */
#if 0
    if (Req.use_cm)
        cm_open(dev);
    else
#endif
        int r = ib_open(dev);
    if (r != 0) {
        return r;
    }

    /* Request CQ notification if not polling */
    if (!Req.poll_mode) {
        if (ibv_req_notify_cq(dev->cq, 0) != 0)
            return error(SYS, "failed to request CQ notification");
    }

    return r;
}


/*
 * Called after rd_open to prepare both ends.
 */
int
rd_prep(const DEVICE *dev, CONNECTION *con)
{
#if 0
	Do this outside the qperf library: perform all rd_mrallocs, do an all2all
	of the generated regions
    /* Set the size of the messages we transfer */
    if (size == 0)
        dev->msg_size = Req.msg_size;

    /* Allocate memory region */
    if (size == 0)
        size = dev->msg_size;
    if (dev->trans == IBV_QPT_UD)
        size += GRH_SIZE;
    rd_mralloc(dev, size);

    /* Exchange node information */
    {
        NODE node;

        enc_init(&node);
        enc_node(&dev->lnode);
        send_mesg(&node, sizeof(node), "node information");
        recv_mesg(&node, sizeof(node), "node information");
        dec_init(&node);
        dec_node(&dev->rnode);
    }
#endif

    /* Second phase of open for devices */
#if 0
	Do this outside the qperf library: migrate all our qp etc
    if (Req.use_cm) 
        cm_prep(dev);
    else
#endif
        int r = ib_prep(dev, con);

    return r;
}


/*
 * Show node information when debugging.
 */
void
show_node_info(const DEVICE *dev, const CONNECTION *con, const REGION *r)
{
    const NODE *n;

    if (!Debug)
        return;
    n = &dev->lnode;
    const CONN_DESCRIPTOR *c = &con->local;

#if 0
    if (Req.use_cm) 
        debug("L: rkey=%08x vaddr=%010x", n->rkey, n->vaddr);
#ifdef HAS_XRC
    else if (dev->trans == IBV_QPT_XRC) {
        debug("L: lid=%04x qpn=%06x psn=%06x rkey=%08x vaddr=%010x srqn=%08x",
                        n->lid, n->qpn, n->psn, n->rkey, n->vaddr, n->srqn);
    }
#endif
    else {
#endif
        debug("L: lid=%04x qpn=%06x psn=%06x rkey=%08x vaddr=%010x",
	      n->lid, c->qpn, c->psn, r->key, r->vaddr);
#if 0
    }
#endif
}

void
show_remote_node_info(const NODE *n, const CONNECTION *con, const REGION *r)
{
    const CONN_DESCRIPTOR *c = &con->remote;
#if 0
    n = &dev->rnode;
    if (Req.use_cm) 
        debug("R: rkey=%08x vaddr=%010x", n->rkey, n->vaddr);
#ifdef HAS_XRC
    else if (dev->trans == IBV_QPT_XRC) {
        debug("R: lid=%04x qpn=%06x psn=%06x rkey=%08x vaddr=%010x srqn=%08x",
                            n->lid, n->qpn, n->psn, n->rkey, n->vaddr);
    }
#endif
    else {
#endif
        debug("R: lid=%04x qpn=%06x psn=%06x rkey=%08x vaddr=%010x",
	      n->lid, c->qpn, c->psn, r->key, r->vaddr, n->srqn);
#if 0
    }
#endif
}


/*
 * Close a RDMA device.  We must destroy the CQ before the QP otherwise the
 * ibv_destroy_qp call seems to sometimes hang.  We must also destroy the QP
 * before destroying the memory region as we cannot destroy the memory region
 * if there are references still outstanding.  Hopefully we now have things in
 * the right order.
 */

int
rd_close_qp(CONNECTION *con)
{
    if (con->qp)
        ibv_destroy_qp(con->qp);

    return 0;
}


int
rd_close(DEVICE *dev)
{
#if 0
    if (Req.use_cm)
        cm_close(dev);
    else
#endif
        ib_close1(dev);

    if (dev->ah)
        ibv_destroy_ah(dev->ah);
    if (dev->cq)
        ibv_destroy_cq(dev->cq);

    return 0;
}


int
rd_close_2(DEVICE *dev)
{
    if (dev->pd)
        ibv_dealloc_pd(dev->pd);
    if (dev->channel)
        ibv_destroy_comp_channel(dev->channel);

#if 0
    if (!Req.use_cm)
#endif
        ib_close2(dev);

    memset(dev, 0, sizeof(*dev));

    return 0;
}


/*
 * Create a queue pair.
 */
int
rd_create_qp(DEVICE *dev,
             CONNECTION *con,
             struct ibv_context *context,
             struct rdma_cm_id *id)
{
    /* Set up and verify rd_atomic parameters */
    {
        struct ibv_device_attr dev_attr;

        if (ibv_query_device(context, &dev_attr) != SUCCESS0)
            return error(SYS, "query device failed");
#if 0
        if (Req.rd_atomic == 0)
            dev->lnode.rd_atomic = dev_attr.max_qp_rd_atom;
        else if (Req.rd_atomic <= dev_attr.max_qp_rd_atom)
            dev->lnode.rd_atomic = Req.rd_atomic;
        else
            return error(0, "device only supports %d (< %d) RDMA reads or atomics",
                                    dev_attr.max_qp_rd_atom, Req.rd_atomic);
#endif
    }

    /* Create queue pair */
    {
        struct ibv_qp_init_attr qp_attr ={
            .send_cq = dev->cq,
            .recv_cq = dev->cq,
            .cap     ={
                .max_send_wr     = dev->max_send_wr,
                .max_recv_wr     = dev->max_recv_wr,
                .max_send_sge    = 1,
                .max_recv_sge    = 1,
            },
            .qp_type = IBV_QPT_RC,
        };

#if 0
        if (Req.use_cm) {
            if (rdma_create_qp(id, dev->pd, &qp_attr) != 0)
                return error(SYS, "failed to create QP");
            dev->qp = id->qp;
        } else {
#ifdef HAS_XRC
            if (dev->trans == IBV_QPT_XRC) {
                struct ibv_srq_init_attr srq_attr ={
                    .attr ={
                        .max_wr  = dev->max_recv_wr,
                        .max_sge = 1
                    }
                };

                dev->xrc = ibv_open_xrc_domain(context, -1, O_CREAT);
                if (!dev->xrc)
                    return error(SYS, "failed to open XRC domain");

                dev->srq = ibv_create_xrc_srq(dev->pd, dev->xrc, dev->cq,
                                                                    &srq_attr);
                if (!dev->srq)
                    return error(SYS, "failed to create SRQ");

                qp_attr.cap.max_recv_wr  = 0;
                qp_attr.cap.max_recv_sge = 0;
                qp_attr.xrc_domain       = dev->xrc;
            }
#endif /* HAS_XRC */
#endif

            con->qp = ibv_create_qp(dev->pd, &qp_attr);
            if (!con->qp)
                return error(SYS, "failed to create QP");
#if 0
        }
#endif
    }

    /* Get QP attributes */
    {
        struct ibv_qp_attr qp_attr;
        struct ibv_qp_init_attr qp_init_attr;

        if (ibv_query_qp(con->qp, &qp_attr, 0, &qp_init_attr) != 0)
            return error(SYS, "query QP failed");
        con->max_inline = qp_attr.cap.max_inline_data;
    }

    return 0;
}


/*
 * Allocate a memory region and register it.  I thought this routine should
 * never be called with a size of 0 as prior code checks for that and sets it
 * to some default value.  I appear to be wrong.  In that case, size is set to
 * 1 so other code does not break.
 */
int
rd_mralloc(REGION *region, const DEVICE *dev, size_t size)
{
    int flags;
    int pagesize;

    if (size == 0)
        size = 1;

    pagesize = sysconf(_SC_PAGESIZE);
    if (posix_memalign((void **)&region->vaddr, pagesize, size) != 0)
	return error(SYS, "failed to allocate memory");
    memset(region->vaddr, 0, size);
    flags = IBV_ACCESS_LOCAL_WRITE  |
	IBV_ACCESS_REMOTE_READ  |
	IBV_ACCESS_REMOTE_WRITE |
	IBV_ACCESS_REMOTE_ATOMIC;
    region->mr = ibv_reg_mr(dev->pd, region->vaddr, size, flags);
    if (!region->mr)
	return error(SYS, "failed to allocate memory region");
    region->key  = region->mr->rkey;
    region->size = size;

    return 0;
}


/*
 * Free the memory region.
 */
int
rd_mrfree(REGION *region, const DEVICE *dev)
{
    if (region->mr)
        ibv_dereg_mr(region->mr);
    region->mr = NULL;

    if (region->vaddr)
        free(region->vaddr);
    region->vaddr = NULL;
    region->size = 0;
    region->key = 0;

    return 0;
}


/*
 * Open an InfiniBand device.
 */
static int
ib_open(DEVICE *dev)
{
    /* Determine MTU */
    {
        int mtu = Req.mtu_size;

        if (mtu == 256)
            dev->ib.mtu = IBV_MTU_256;
        else if (mtu == 512)
            dev->ib.mtu = IBV_MTU_512;
        else if (mtu == 1024)
            dev->ib.mtu = IBV_MTU_1024;
        else if (mtu == 2048)
            dev->ib.mtu = IBV_MTU_2048;
        else if (mtu == 4096)
            dev->ib.mtu = IBV_MTU_4096;
        else
            return error(0, "bad MTU: %d; must be 256/512/1K/2K/4K", mtu);
    }

    /* Determine port */
    {
        int port = 1;
        char *p = strchr(Req.id, ':');

        if (p) {
            *p++ = '\0';
            port = atoi(p);
            if (port < 1)
                return error(0, "bad IB port: %d; must be at least 1", port);
        }
        dev->ib.port = port;
    }

    /* Determine static rate */
    {
        RATES *q = Rates;
        RATES *r = q + cardof(Rates);

        for (;; ++q) {
            if (q >= r)
                return error(SYS, "bad static rate: %s", Req.static_rate);
            if (strcmp(Req.static_rate, q->name) == 0) {
                dev->ib.rate = q->rate;
                break;
            }
        }
    }

#if 0
    // Only for UC
    /* Set up Q Key */
    dev->qkey = QKEY;
#endif

    /* Open device */
    {
        struct ibv_device *device;
        const char *name = Req.id[0] ? Req.id : 0;

        dev->ib.devlist = ibv_get_device_list(0);
        if (!dev->ib.devlist)
            return error(SYS, "failed to find any InfiniBand devices");
        if (!name)
            device = *dev->ib.devlist;
        else {
            struct ibv_device **d = dev->ib.devlist;
            while ((device = *d++))
                if (strcmp(ibv_get_device_name(device), name) == 0)
                    break;
        }
        if (!device)
            return error(SYS, "failed to find InfiniBand device");
        dev->ib.context = ibv_open_device(device);
        if (!dev->ib.context) {
            const char *s = ibv_get_device_name(device);
            return error(SYS, "failed to open device %s", s);
        }
    }

    /* Set up local node LID */
    {
        struct ibv_port_attr port_attr;
        int stat = ibv_query_port(dev->ib.context, dev->ib.port, &port_attr);

        if (stat != 0)
            return error(SYS, "query port failed");
        srand(getpid()*time(0));
        dev->lnode.lid = port_attr.lid;
	if (port_attr.lmc > 0)
	    dev->lnode.lid += Req.src_path_bits & ((1 << port_attr.lmc) - 1);
    }

    if (!Req.poll_mode) {
        /* Allocate completion channel */
        dev->channel = ibv_create_comp_channel(dev->ib.context);
        if (!dev->channel)
            return error(SYS, "failed to create completion channel");
    } else {
        fprintf(stderr, "skip the completion channel\n");
    }

    /* Allocate protection domain */
    dev->pd = ibv_alloc_pd(dev->ib.context);
    if (!dev->pd)
        return error(SYS, "failed to allocate protection domain");

    /* Create completion queue */
    dev->cq = ibv_create_cq(dev->ib.context,
                        dev->max_send_wr+dev->max_recv_wr, 0, dev->channel, 0);
    if (!dev->cq)
        return error(SYS, "failed to create completion queue");

    /* Set up alternate port LID */
    if (Req.alt_port) {
        struct ibv_port_attr port_attr;
        int stat = ibv_query_port(dev->ib.context, Req.alt_port, &port_attr);

        if (stat != SUCCESS0)
            return error(SYS, "query port failed");
        dev->lnode.alt_lid = port_attr.lid;
	if (port_attr.lmc > 0)
	    dev->lnode.alt_lid +=
                Req.src_path_bits & ((1 << port_attr.lmc) - 1);
    }

    return 0;
}


#ifdef UNUSED
static int
ib_create_qp(CONNECTION *con, DEVICE *dev, const NODE *peer)
{
    /* Create QP */
    return rd_create_qp(con, dev, dev->ib.context, 0);
}
#endif


int
rd_open_2(const DEVICE *dev, CONNECTION *con)
{
    /* Modify queue pair to INIT state */
    {
        struct ibv_qp_attr attr ={
            .qp_state       = IBV_QPS_INIT,
            .pkey_index     = 0,
            .port_num       = dev->ib.port
        };
        int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;

#if 0
        if (dev->trans == IBV_QPT_UD) {
            flags |= IBV_QP_QKEY;
            attr.qkey = dev->qkey;
#ifdef HAS_XRC
        } else if (dev->trans == IBV_QPT_RC || dev->trans == IBV_QPT_XRC) {
#else
        } else if (dev->trans == IBV_QPT_RC) {
#endif
#endif
            flags |= IBV_QP_ACCESS_FLAGS;
            attr.qp_access_flags =
                IBV_ACCESS_REMOTE_READ  |
                IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_ATOMIC;
#if 0
        } else if (dev->trans == IBV_QPT_UC) {
            flags |= IBV_QP_ACCESS_FLAGS;
            attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        }
#endif
        if (ibv_modify_qp(con->qp, &attr, flags) != SUCCESS0)
            return error(SYS, "failed to modify QP to INIT state");
    }

    /* Set up local node QP number, PSN and SRQ number */
    con->local.qpn = con->qp->qp_num;
    con->local.psn = rand() & 0xffffff;
#ifdef HAS_XRC
    if (dev->trans == IBV_QPT_XRC)
        dev->lnode.srqn = dev->srq->xrc_srq_num;
#endif

    return 0;
}


/*
 * Prepare the InfiniBand connection for receiving and sending.  Final stage of
 * open.
 */
static int
ib_prep(const DEVICE *dev, CONNECTION *con)
{
    int flags;
    struct ibv_qp_attr rtr_attr ={
        .qp_state           = IBV_QPS_RTR,
        .path_mtu           = dev->ib.mtu,
        .dest_qp_num        = con->remote.qpn,
        .rq_psn             = con->remote.psn,
        .min_rnr_timer      = MIN_RNR_TIMER,
        .max_dest_rd_atomic = 1,	// con->lnode.rd_atomic,
        .ah_attr            = {
            .dlid           = con->rnode.lid,
            .port_num       = dev->ib.port,
            .static_rate    = dev->ib.rate,
	    .src_path_bits  = 0,	// Req.src_path_bits,
            .sl             = 0,	// Req.sl
        }
    };
    struct ibv_qp_attr rts_attr ={
        .qp_state          = IBV_QPS_RTS,
        .timeout           = LOCAL_ACK_TIMEOUT,
        .retry_cnt         = RETRY_CNT,
        .rnr_retry         = RNR_RETRY_CNT,
        .sq_psn            = con->local.psn,
        .max_rd_atomic     = 1,	// dev->rnode.rd_atomic,
        .path_mig_state    = IBV_MIG_REARM,
        .alt_port_num      = Req.alt_port,
        .alt_ah_attr       = {
            .dlid          = con->rnode.alt_lid,
            .port_num      = Req.alt_port,
            .static_rate   = dev->ib.rate,
	    .src_path_bits = Req.src_path_bits,
            .sl            = Req.sl
        }
    };
#if 0
    struct ibv_ah_attr ah_attr ={
        .dlid          = dev->rnode.lid,
        .port_num      = dev->ib.port,
        .static_rate   = dev->ib.rate,
	.src_path_bits = Req.src_path_bits,
        .sl            = Req.sl
    };
#endif

#if 0
    if (dev->trans == IBV_QPT_UD) {
        /* Modify queue pair to RTR */
        flags = IBV_QP_STATE;
        if (ibv_modify_qp(dev->qp, &rtr_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTR");

        /* Modify queue pair to RTS */
        flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
        if (ibv_modify_qp(dev->qp, &rts_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTS");

        /* Create address handle */
        dev->ah = ibv_create_ah(dev->pd, &ah_attr);
        if (!dev->ah)
            return error(SYS, "failed to create address handle");
#ifdef HAS_XRC
    } else if (dev->trans == IBV_QPT_RC || dev->trans == IBV_QPT_XRC) {
#else
    } else if (dev->trans == IBV_QPT_RC) {
#endif
#endif
        /* Modify queue pair to RTR */
        flags = IBV_QP_STATE              |
                IBV_QP_AV                 |
                IBV_QP_PATH_MTU           |
                IBV_QP_DEST_QPN           |
                IBV_QP_RQ_PSN             |
                IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER;
        if (ibv_modify_qp(con->qp, &rtr_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTR");

        /* Modify queue pair to RTS */
        flags = IBV_QP_STATE     |
                IBV_QP_TIMEOUT   |
                IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN    |
                IBV_QP_MAX_QP_RD_ATOMIC;
        if (
#if 0
            dev->trans == IBV_QPT_RC &&
#endif
            con->rnode.alt_lid)
            flags |= IBV_QP_ALT_PATH | IBV_QP_PATH_MIG_STATE;
        if (ibv_modify_qp(con->qp, &rts_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTS");
#if 0
    } else if (dev->trans == IBV_QPT_UC) {
        /* Modify queue pair to RTR */
        flags = IBV_QP_STATE    |
                IBV_QP_AV       |
                IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN;
        if (ibv_modify_qp(dev->qp, &rtr_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTR");

        /* Modify queue pair to RTS */
        flags = IBV_QP_STATE |
                IBV_QP_SQ_PSN;
        if (dev->rnode.alt_lid)
            flags |= IBV_QP_ALT_PATH | IBV_QP_PATH_MIG_STATE;
        if (ibv_modify_qp(dev->qp, &rts_attr, flags) != 0)
            return error(SYS, "failed to modify QP to RTS");
    }
#endif

    return 0;
}


/*
 * Close an InfiniBand device, part 1.
 */
static int
ib_close1(DEVICE *dev)
{
    if (dev->srq)
        ibv_destroy_srq(dev->srq);
#ifdef HAS_XRC
    if (dev->xrc)
        ibv_close_xrc_domain(dev->xrc);
#endif

    return 0;
}


/*
 * Close an InfiniBand device, part 2.
 */
static int
ib_close2(DEVICE *dev)
{
    if (dev->ib.context)
        ibv_close_device(dev->ib.context);
    if (dev->ib.devlist)
        free(dev->ib.devlist);

    return 0;
}


/*
 * Post 1 RDMA request n times.
 */
int
rd_post_rdma_std_1(CONNECTION *con, uint32_t lkey, uint32_t rkey,
                 int opcode,
		 size_t n,
		 const void *local_addr,
		 const void *remote_addr,
		 const size_t sizes)
{
    struct ibv_sge sge ={
        .lkey   = lkey,
	.length = sizes,
	.addr   = (uint64_t)local_addr,
    };
    struct ibv_send_wr wr ={
        .wr_id      = WRID_RDMA,
        .sg_list    = &sge,
        .num_sge    = 1,
        .opcode     = opcode,
        .send_flags = IBV_SEND_SIGNALED,
        .wr = {
            .rdma = {
                .remote_addr = (uint64_t)remote_addr,
                .rkey        = rkey,
            }
        },
    };

    errno = 0;
    for (size_t i = 0; i < n; i++) {
	struct ibv_send_wr *badwr;

        if (opcode != IBV_WR_RDMA_READ && sizes <= con->max_inline)
            wr.send_flags |= IBV_SEND_INLINE;
        if (ibv_post_send(con->qp, &wr, &badwr) != SUCCESS0) {
            return error(SYS, "failed to post %s", opcode_name(wr.opcode));
        }
#if 0
        if (opcode != IBV_WR_RDMA_READ) {
            LStat.s.no_bytes += sizes[i];
            LStat.s.no_msgs++;
        }
#endif
    }

    return 0;
}


/*
 * Post n RDMA requests.
 */
int
rd_post_rdma_std(CONNECTION *con, uint32_t lkey, uint32_t rkey,
                 int opcode,
		 size_t n,
		 const void **local_addr,
		 const void **remote_addr,
		 const size_t *sizes)
{
    struct ibv_sge sge ={
        .lkey   = lkey,
    };
    struct ibv_send_wr wr ={
        .wr_id      = WRID_RDMA,
        .sg_list    = &sge,
        .num_sge    = 1,
        .opcode     = opcode,
        .send_flags = IBV_SEND_SIGNALED,
        .wr = {
            .rdma = {
                .rkey        = rkey,
            }
        },
    };

    errno = 0;
    for (size_t i = 0; i < n; i++) {
	struct ibv_send_wr *badwr;

	sge.length = sizes[i];
	sge.addr   = (uint64_t)local_addr[i];
	wr.wr.rdma.remote_addr = (uint64_t)remote_addr[i];
        if (opcode != IBV_WR_RDMA_READ && sizes[i] <= con->max_inline)
            wr.send_flags |= IBV_SEND_INLINE;
        if (ibv_post_send(con->qp, &wr, &badwr) != SUCCESS0) {
            return error(SYS, "failed to post %s", opcode_name(wr.opcode));
        }
#if 0
        if (opcode != IBV_WR_RDMA_READ) {
            LStat.s.no_bytes += sizes[i];
            LStat.s.no_msgs++;
        }
#endif
    }

    return 0;
}


/*
 * Poll the completion queue.
 */
int
rd_poll(DEVICE *dev, struct ibv_wc *wc, int nwc)
{
    int n;

    if (!Req.poll_mode
#if 0
        && !Finished
#endif
        ) {
        void *ectx;
        struct ibv_cq *ecq;

        if (ibv_get_cq_event(dev->channel, &ecq, &ectx) != SUCCESS0)
            return maybe(0, "failed to get CQ event");
        if (ecq != dev->cq)
            return error(0, "CQ event for unknown CQ");
        if (ibv_req_notify_cq(dev->cq, 0) != SUCCESS0)
            return maybe(0, "failed to request CQ notification");
	ibv_ack_cq_events(dev->cq, 1);
    }
    n = ibv_poll_cq(dev->cq, nwc, wc);
    if (n < 0)
        return maybe(0, "CQ poll failed");
    return n;
}


/*
 * We encountered an error in a system call which might simply have been
 * interrupted by the alarm that signaled completion of the test.  Generate the
 * error if appropriate or return the requested value.  Final return is just to
 * silence the compiler.
 */
int
maybe(int val, char *msg)
{
    if (errno == EINTR)
        return val;
    return error(SYS, msg);
}


#if 0
/*
 * Encode a NODE structure into a data stream.
 */
void
enc_node(NODE *host)
{
    enc_int(host->vaddr,     sizeof(host->vaddr));
    enc_int(host->lid,       sizeof(host->lid));
    enc_int(host->qpn,       sizeof(host->qpn));
    enc_int(host->psn,       sizeof(host->psn));
    enc_int(host->srqn,      sizeof(host->srqn));
    enc_int(host->rkey,      sizeof(host->rkey));
    enc_int(host->alt_lid,   sizeof(host->alt_lid));
    enc_int(host->rd_atomic, sizeof(host->rd_atomic));
}


/*
 * Decode a NODE structure from a data stream.
 */
void
dec_node(NODE *host)
{
    host->vaddr     = dec_int(sizeof(host->vaddr));
    host->lid       = dec_int(sizeof(host->lid));
    host->qpn       = dec_int(sizeof(host->qpn));
    host->psn       = dec_int(sizeof(host->psn));
    host->srqn      = dec_int(sizeof(host->srqn));
    host->rkey      = dec_int(sizeof(host->rkey));
    host->alt_lid   = dec_int(sizeof(host->alt_lid));
    host->rd_atomic = dec_int(sizeof(host->rd_atomic));
}
#endif


/*
 * Handle a CQ error and return true if it is recoverable.
 */
int
do_error(int status, uint64_t *errors)
{
    ++*errors;
    return cq_error(status);
}


int
cq_error(int status)
{
    for (size_t i = 0; i < cardof(CQErrors); ++i) {
	if (CQErrors[i].value == status)
	    return error(0, "%s failed: %s", "qprf-rdma", CQErrors[i].name);
    }
    return error(0, "%s failed: CQ error %d", "qprf-rdma", status);
}



/*
 * Return the name of an opcode.
 */
char *
opcode_name(int opcode)
{
    size_t i;

    for (i = 0; i < cardof(Opcodes); ++i)
        if (Opcodes[i].value == opcode)
            return Opcodes[i].name;
    return "unknown operation";
}
