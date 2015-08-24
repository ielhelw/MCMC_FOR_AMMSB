#ifndef QPERF_RDMA_H__
#define QPERF_RDMA_H__

#ifdef __cplusplus
extern "C" {
#endif

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

// #include <fcntl.h>
// #include <errno.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <unistd.h>
// #include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
// #include "qperf.h"

/*
 * RDMA parameters.
 */
#define QKEY                0x11111111  /* Q_Key */
#define NCQE                1024        /* Number of CQ entries */
#define GRH_SIZE            40          /* InfiniBand GRH size */
#define MTU_SIZE            2048        /* Default MTU Size */
#define RETRY_CNT           7           /* RC retry count */
#define RNR_RETRY_CNT       7           /* RC RNR retry count */
#define MIN_RNR_TIMER       12          /* RC Minimum RNR timer */
#define LOCAL_ACK_TIMEOUT   14          /* RC local ACK timeout */


/*
 * Work request IDs.
 */
#define WRID_SEND   1                   /* Send */
#define WRID_RECV   2                   /* Receive */
#define WRID_RDMA   3                   /* RDMA */


/*
 * Constants.
 */
#define K2      (2*1024)
#define K64     (64*1024)


/*
 * For convenience.
 */
typedef enum ibv_wr_opcode ibv_op;
typedef struct ibv_comp_channel ibv_cc;



/*
 * InfiniBand specific information.
 */
typedef struct IBINFO {
    int                 mtu;            /* MTU */
    int                 port;           /* Port */
    int                 rate;           /* Static rate */
    struct ibv_context *context;        /* Context */
    struct ibv_device **devlist;        /* Device list */
} IBINFO;


typedef struct NODE {
    uint32_t    lid;                    /* Local ID */
    uint32_t    srqn;                   /* Shared queue number */
    uint32_t    alt_lid;                /* Alternate Path Local LID */
} NODE;


typedef struct CONN_DESCRIPTOR {
    int32_t     qpn;                    /* Queue pair number */
    int32_t     psn;                    /* Packet sequence number */
    int32_t     rd_atomic;
} CONN_DESCRIPTOR;


/*
 * RDMA device descriptor.
 */
typedef struct DEVICE {
    NODE             lnode;             /* Local node information */
    IBINFO           ib;                /* InfiniBand information */
#if 0
    CMINFO           cm;                /* Connection Manager information */
    uint32_t         qkey;              /* Q Key for UD */
    int              trans;             /* QP transport */
    int              msg_size;          /* Message size */
    int              buf_size;          /* Buffer size */
#endif
    int              max_send_wr;       /* Maximum send work requests */
    int              max_recv_wr;       /* Maximum receive work requests */
#if 0
    char            *buffer;            /* Buffer */
#endif
    ibv_cc          *channel;           /* Channel */
    struct ibv_pd   *pd;                /* Protection domain */
    struct ibv_cq   *cq;                /* Completion queue */
    struct ibv_ah   *ah;                /* Address handle */
    struct ibv_srq  *srq;               /* Shared receive queue */
#ifdef HAS_XRC
    ibv_xrc         *xrc;               /* XRC domain */
#endif
} DEVICE;


/**
 * QP wrapper
 */
typedef struct CONNECTION {
    CONN_DESCRIPTOR	local;             /* Local connection information */
    CONN_DESCRIPTOR	remote;            /* Remote connection information */
    NODE                rnode;             /* Remote node information */
    struct ibv_qp      *qp;                /* Queue Pair */
    size_t		max_inline;        /* Maximum amount of inline data */
} CONNECTION;


typedef struct REGION {
    struct ibv_mr      *mr;             /* Memory region */
    uint32_t		key;		/* local region key */
    size_t		size;		/* local region size */
    void	       *vaddr;		/* local region virtual address */
} REGION;


/*
 * Names associated with a value.
 */
typedef struct NAMES {
    int     value;                       /* Value */
    char    *name;                       /* Name */
} NAMES;


/*
 * RDMA speeds and names.
 */
typedef struct RATES {
    const char *name;                   /* Name */
    uint32_t    rate;                   /* Rate */
} RATES;


typedef struct Options {
    int32_t		mtu_size;
    const char	       *id;
    int32_t             port;
    int32_t             rd_atomic;
    const char	       *static_rate;
    int32_t		src_path_bits;
    int32_t		sl;
    int32_t             poll_mode;
    int32_t             alt_port;
} Options;

extern Options Req;

/*
 * Function prototypes.
 */
const char *rd_get_error_message(void);
void debug(char *fmt, ...);

int     cq_error(int status);
int     do_error(int status, uint64_t *errors);

int     rd_open(DEVICE *dev, int trans, int max_send_wr, int max_recv_wr);
int     rd_prep(const DEVICE *dev, CONNECTION *con);
int     rd_open_2(const DEVICE *dev, CONNECTION *con);
int     rd_create_qp(DEVICE *dev,
                     CONNECTION *con,
                     struct ibv_context *context,
                     struct rdma_cm_id *id);
int     rd_close_qp(CONNECTION *con);
int     rd_close(DEVICE *dev);
int     rd_close_2(DEVICE *dev);
int	rd_client_rdma_bw(DEVICE *dev,
			  CONNECTION *con,
			  size_t n_req,   
			  const void **local_addr,
			  const void **remote_addr,
			  const size_t *sizes);
int     rd_mralloc(REGION *region, const DEVICE *dev, size_t size);
int     rd_mrfree(REGION *region, const DEVICE *dev);
int     rd_poll(DEVICE *dev, struct ibv_wc *wc, int nwc);
int     rd_post_rdma_std_1(CONNECTION *con, uint32_t lkey, uint32_t rkey,
                           int opcode,
                           size_t n,
                           const void* local_addr,
                           const void* remote_addr,
                           const size_t sizes);
int     rd_post_rdma_std(CONNECTION *con, uint32_t lkey, uint32_t rkey,
                         int opcode,
			 size_t n,
			 void* const* local_addr,
			 const void* const *remote_addr,
			 const size_t *sizes);

int      maybe(int val, char *msg);
char    *opcode_name(int opcode);
void     show_node_info(const DEVICE *dev, const CONNECTION *con, const REGION *r);
void    show_remote_node_info(const NODE *n, const CONNECTION *con, const REGION *r);
#if 0
void	enc_node(NODE *host);
void	dec_node(NODE *host);
#endif


/*
 * This routine is never called and is solely to avoid compiler warnings for
 * functions that are not currently being used.
 */
void    rdma_not_called(void);

#ifdef __cplusplus
} 	// extern
#endif

#endif	// ndef QPERF_RDMA_H__
